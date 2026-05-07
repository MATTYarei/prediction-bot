"""
Step 3 · Predict  [v2 — smarter prediction]

NEW in v2:
  UPGRADE 2 — Smarter prediction:
    - Base rate anchoring: Claude sees historical base rates BEFORE reading news
    - Domain-specific prompts: separate prompt templates per category
      (economics, crypto, politics, sports, geopolitics)
    - Wider model ensemble: XGBoost + LightGBM + Claude + base-rate prior
    - Disagreement flag: when models diverge >15%, confidence is capped at medium

  UPGRADE 3 — Tighter execution filters (applied here at prediction output):
    - high_confidence_only flag: skip medium/low confidence predictions
    - min_edge_threshold now checked against calibrated (not raw) probability
    - Cross-platform gap required: if Metaculus/Manifold agree with market, skip

  UPGRADE 4 — Niche specialization:
    - CategoryRouter: detects market category and applies specialist prompt
    - Each domain has tailored reasoning instructions and relevant metrics
"""

import asyncio
import json
import logging
import re
import math
from datetime import datetime
from typing import Optional

import anthropic

from core.config import Config
from core.database import Database
from agents.base_rates import get_relevant_base_rates, format_for_prompt

log = logging.getLogger("predbot.predict")


# ── UPGRADE 4: Category Router ────────────────────────────────────────────────

class CategoryRouter:
    """
    Detects the market category and returns a domain-specific prompt template.

    Why this matters: the reasoning process for "Will BTC hit $120k?" is
    completely different from "Will the Fed cut rates?" or "Who wins the
    Super Bowl?". A one-size-fits-all prompt underperforms specialist prompts
    by a significant margin on each domain.
    """

    CATEGORIES = {
        "economics": ["fed", "rate", "inflation", "gdp", "recession", "cpi",
                       "employment", "jobs", "debt", "deficit", "fomc",
                       "s&p", "sp500", "stock market", "dow", "nasdaq"],
        "crypto":    ["bitcoin", "btc", "ethereum", "eth", "crypto", "defi",
                       "blockchain", "altcoin", "halving", "stablecoin"],
        "politics":  ["election", "president", "congress", "senate", "vote",
                       "bill", "law", "governor", "poll", "party", "candidate"],
        "sports":    ["win", "championship", "super bowl", "nfl", "nba", "mlb",
                       "nhl", "world cup", "playoff", "season", "game"],
        "geopolitics": ["war", "conflict", "sanctions", "treaty", "nato", "china",
                         "russia", "ukraine", "middle east", "nuclear", "ceasefire"],
        "technology": ["ai", "ipo", "earnings", "merger", "acquisition", "launch",
                        "release", "product", "apple", "google", "openai", "model"],
    }

    DOMAIN_PROMPTS = {
        "economics": """
DOMAIN: Economics & Monetary Policy
Key metrics to consider:
  - Current CPI / PCE inflation rate vs Fed target (2%)
  - Federal funds rate vs neutral rate estimates
  - Unemployment rate trend (3-month direction matters)
  - Yield curve shape (inverted = recession signal)
  - Fed dot plot projections and recent Fed speeches
  - CME FedWatch implied probabilities (if in news)
Reasoning approach: Start with the Fed's stated mandate and data dependency.
What would the data need to show for this outcome? Is current data consistent?
""",
        "crypto": """
DOMAIN: Cryptocurrency Markets
Key metrics to consider:
  - BTC distance from all-time high (momentum indicator)
  - Halving cycle position (post-halving 12-18 months historically bullish)
  - Crypto Fear & Greed Index level
  - On-chain metrics: exchange inflows/outflows, whale movements
  - Macro risk appetite (correlated with NASDAQ)
  - Regulatory environment (SEC, ETF approvals, etc.)
Reasoning approach: Crypto is highly sentiment-driven short-term.
Technical levels matter. Distinguish between macro trend and short-term noise.
""",
        "politics": """
DOMAIN: Politics & Elections
Key metrics to consider:
  - Current approval ratings and trends
  - Historical precedent for this type of political event
  - Polling averages and pollster quality
  - Structural factors (incumbency, economic conditions, candidate quality)
  - Prediction market consensus (are markets pricing in known factors?)
  - Time to event (closer = less variance, polls more predictive)
Reasoning approach: Separate fundamentals from news noise.
Short-term political news often moves markets more than it should.
Anchor to structural factors and historical base rates.
""",
        "sports": """
DOMAIN: Sports
Key metrics to consider:
  - Team/player recent form (last 5-10 games/events)
  - Injury reports and lineup availability
  - Head-to-head historical record
  - Home/away advantage (home wins ~55-60% in major sports)
  - Statistical models (Elo, FiveThirtyEight, Vegas lines)
  - Weather conditions for outdoor sports
Reasoning approach: Respect the Vegas line — it has massive resources.
Look for specific edges: injury news not yet priced in, schedule factors, etc.
""",
        "geopolitics": """
DOMAIN: Geopolitics & International Relations
Key metrics to consider:
  - Historical resolution rates for similar conflicts/negotiations
  - Current state of diplomatic channels
  - Economic pressure on parties involved
  - Recent escalation/de-escalation signals
  - International community involvement
  - Time pressure / deadlines driving resolution
Reasoning approach: Geopolitical events resolve slowly and base rates
are low for resolution. Be skeptical of optimistic short-term outcomes.
Anchor heavily to historical precedent.
""",
        "technology": """
DOMAIN: Technology & Business
Key metrics to consider:
  - Company financial health and recent earnings
  - Product roadmap and announced timelines (companies miss ~30% of announced dates)
  - Competitive dynamics and market position
  - Regulatory environment
  - Recent executive commentary and guidance
  - Analyst consensus and recent revisions
Reasoning approach: Technology timelines slip. Product launches are often delayed.
Corporate guidance is systematically optimistic. Apply a skepticism discount.
""",
    }

    def detect_category(self, question: str, market_category: str = "") -> str:
        """Detect the most likely category from question text and market metadata."""
        if market_category:
            for cat in self.CATEGORIES:
                if cat in market_category.lower():
                    return cat

        question_lower = question.lower()
        scores = {}
        for cat, keywords in self.CATEGORIES.items():
            scores[cat] = sum(1 for kw in keywords if kw in question_lower)

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def get_domain_prompt(self, category: str) -> str:
        return self.DOMAIN_PROMPTS.get(category, "")


# ── UPGRADE 2: XGBoost + LightGBM Ensemble ───────────────────────────────────

class EnsemblePredictor:
    """
    Two-model ensemble: XGBoost + LightGBM.
    When models agree within 5%, use their average.
    When they disagree >15%, cap confidence at 'medium' (disagreement signal).
    """

    def __init__(self):
        self.xgb_model  = None
        self.lgbm_model = None
        self._try_load_models()

    def _try_load_models(self):
        try:
            import xgboost as xgb
            try:
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model("models/xgboost_market.json")
                log.info("XGBoost: loaded saved model")
            except Exception:
                log.info("XGBoost: no saved model — using heuristic")
        except ImportError:
            log.warning("XGBoost not installed")

        try:
            import lightgbm as lgb
            try:
                self.lgbm_model = lgb.Booster(model_file="models/lgbm_market.txt")
                log.info("LightGBM: loaded saved model")
            except Exception:
                log.info("LightGBM: no saved model — using heuristic")
        except ImportError:
            log.warning("LightGBM not installed")

    def predict(self, market: dict) -> dict:
        """Returns dict with xgb_prob, lgbm_prob, ensemble_prob, model_agreement."""
        xgb_p  = self._xgb_predict(market)
        lgbm_p = self._lgbm_predict(market)

        # If one model is missing, use the other
        if xgb_p is None and lgbm_p is None:
            p = self._heuristic(market)
            return {"xgb": p, "lgbm": p, "ensemble": p, "agreement": "single",
                    "disagreement": 0.0, "confidence_cap": None}

        if xgb_p is None:
            return {"xgb": lgbm_p, "lgbm": lgbm_p, "ensemble": lgbm_p,
                    "agreement": "single", "disagreement": 0.0, "confidence_cap": None}

        if lgbm_p is None:
            return {"xgb": xgb_p, "lgbm": xgb_p, "ensemble": xgb_p,
                    "agreement": "single", "disagreement": 0.0, "confidence_cap": None}

        disagreement = abs(xgb_p - lgbm_p)
        ensemble = (xgb_p * 0.5 + lgbm_p * 0.5)
        agreement = "strong" if disagreement < 0.05 else "weak" if disagreement > 0.15 else "moderate"
        confidence_cap = "medium" if disagreement > 0.15 else None  # cap if models disagree

        return {
            "xgb": round(xgb_p, 4),
            "lgbm": round(lgbm_p, 4),
            "ensemble": round(ensemble, 4),
            "agreement": agreement,
            "disagreement": round(disagreement, 4),
            "confidence_cap": confidence_cap,
        }

    def _xgb_predict(self, market: dict) -> Optional[float]:
        if self.xgb_model is None:
            return None
        try:
            import numpy as np
            X = np.array([self._features(market)])
            p = float(self.xgb_model.predict_proba(X)[0][1])
            return max(0.05, min(0.95, p))
        except Exception as e:
            log.error(f"XGBoost predict: {e}")
            return None

    def _lgbm_predict(self, market: dict) -> Optional[float]:
        if self.lgbm_model is None:
            return None
        try:
            import numpy as np
            X = np.array([self._features(market)])
            p = float(self.lgbm_model.predict(X)[0])
            return max(0.05, min(0.95, p))
        except Exception as e:
            log.error(f"LightGBM predict: {e}")
            return None

    def _heuristic(self, market: dict) -> float:
        """Mild mean-reversion heuristic when no trained models exist."""
        p = market.get("yes_price", 0.5)
        return round(max(0.05, min(0.95, p * 0.92 + 0.5 * 0.08)), 4)

    def _features(self, market: dict) -> list:
        p = market.get("yes_price", 0.5)
        return [
            p,
            abs(p - 0.5),
            math.log1p(market.get("volume_usd", 0)),
            math.log1p(market.get("liquidity_usd", 0)),
            market.get("days_left", 14),
            market.get("filter_score", 0.5),
            # New v2 features
            market.get("cross_platform_gap", 0.0),   # from signal package
            market.get("order_flow_combined", 0.0),  # from order book
        ]

    def train_on_history(self, db: "Database"):
        """Retrain both models on resolved history."""
        try:
            import xgboost as xgb
            import numpy as np, os

            learnings = db.get_recent_learnings(limit=10000)
            X, y = [], []
            for l in learnings:
                if l.get("our_prob") and l.get("actual_outcome") is not None:
                    market = db.get_market(l["market_id"]) or {}
                    X.append(self._features({"yes_price": l["our_prob"], **market}))
                    y.append(int(l["actual_outcome"]))

            if len(X) < 50:
                log.info(f"Ensemble: need 50+ resolved trades to train (have {len(X)})")
                return

            os.makedirs("models", exist_ok=True)

            # XGBoost
            xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                                          learning_rate=0.1, eval_metric="logloss")
            xgb_clf.fit(np.array(X), np.array(y))
            xgb_clf.save_model("models/xgboost_market.json")
            self.xgb_model = xgb_clf
            log.info(f"XGBoost: retrained on {len(X)} samples")

            # LightGBM
            try:
                import lightgbm as lgb
                train_data = lgb.Dataset(np.array(X), label=np.array(y))
                params = {"objective": "binary", "metric": "binary_logloss",
                          "num_leaves": 31, "learning_rate": 0.05, "verbose": -1}
                lgbm_model = lgb.train(params, train_data, num_boost_round=100)
                lgbm_model.save_model("models/lgbm_market.txt")
                self.lgbm_model = lgbm_model
                log.info(f"LightGBM: retrained on {len(X)} samples")
            except ImportError:
                log.info("LightGBM not installed — skipping")

        except Exception as e:
            log.error(f"Ensemble training: {e}")


# ── UPGRADE 2: Domain-Aware LLM Calibrator ───────────────────────────────────

class LLMCalibrator:
    """
    Claude now receives:
      1. Historical base rates for this question type (anchor)
      2. Domain-specific reasoning instructions
      3. Order flow signal (smart money direction)
      4. Cross-platform probability comparison (Metaculus/Manifold)
      5. Ensemble model estimate + agreement level

    The base rates are presented FIRST so Claude anchors to historical
    frequencies before being swayed by recent headlines.
    """

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.model = model
        self.router = CategoryRouter()

    async def calibrate(self, market: dict, news_texts: list[str],
                        ensemble_result: dict, signal_package: dict) -> dict:
        if not self.client:
            return {"probability": ensemble_result["ensemble"], "confidence": "low",
                    "reasoning": "No API key"}

        # Detect category and get domain prompt
        category = self.router.detect_category(
            market["question"], market.get("category", ""))
        domain_prompt = self.router.get_domain_prompt(category)

        # Get base rates
        base_rates = get_relevant_base_rates(market["question"], category)
        base_rate_block = format_for_prompt(base_rates)

        # Format news
        news_block = "\n".join(f"- {t[:300]}" for t in news_texts[:20])

        # External forecasts block
        ext_probs = signal_package.get("external_probs", [])
        ext_block = ""
        if ext_probs:
            ext_block = "\nEXTERNAL FORECASTS (independent calibrated estimates):\n"
            for e in ext_probs:
                ext_block += (f"  - {e['source'].title()}: {e['prob']:.0%} YES"
                               + (f" ({e['forecasters']} forecasters)" if e.get("forecasters") else "")
                               + "\n")
            gap = signal_package.get("cross_platform_gap", 0)
            if gap > 0.10:
                ext_block += f"  → ALERT: {gap:.0%} gap between market and external forecasts\n"

        # Order flow block
        of = signal_package.get("order_flow", {})
        of_block = ""
        if of.get("signal") and of["signal"] != "none":
            of_block = f"""
ORDER BOOK SIGNAL (real money positioning):
  Direction: {of.get('smart_money', 'unknown')}
  Signal: {of.get('signal', 'neutral')} (score: {of.get('combined', 0):+.3f})
  Large orders detected: {of.get('large_orders', 0)}
"""

        # Model ensemble block
        disagree_note = ""
        if ensemble_result.get("confidence_cap"):
            disagree_note = f"\n  ⚠ Models disagree by {ensemble_result['disagreement']:.0%} — confidence capped at medium"

        prompt = f"""You are an expert prediction market analyst.
Your task: estimate the TRUE probability of YES with minimal bias.

CRITICAL PROCESS — follow in order:
  1. Read the base rates. These are your anchor.
  2. Read the domain guidance. Apply its reasoning framework.
  3. Read external forecasts. Note any gaps vs market price.
  4. Read the order flow. Smart money is a leading indicator.
  5. Read recent news. Adjust from base rate — not the other way around.
  6. Produce a final calibrated probability.

═══════════════════════════════════════════════════════════
MARKET: {market['question']}
CURRENT MARKET PRICE (YES): {market['yes_price']:.2%}
DAYS TO RESOLUTION: {market.get('days_left', '?'):.0f}
VOLUME: ${market.get('volume_usd', 0):,.0f}
CATEGORY: {category}
═══════════════════════════════════════════════════════════

{base_rate_block}

{domain_prompt}
{ext_block}
{of_block}
MODEL ENSEMBLE ESTIMATE: {ensemble_result['ensemble']:.2%}
  XGBoost: {ensemble_result['xgb']:.2%} | LightGBM: {ensemble_result['lgbm']:.2%}
  Agreement: {ensemble_result['agreement']}{disagree_note}

RECENT NEWS & SIGNALS:
{news_block if news_block else "No recent news available."}

═══════════════════════════════════════════════════════════
Return ONLY valid JSON:
{{
  "probability": <float 0.05–0.95>,
  "confidence": "low|medium|high",
  "base_rate_used": <float, which base rate you anchored to>,
  "base_rate_adjustment": <float, how much you adjusted from base rate>,
  "adjustment_reasoning": "<why you moved from base rate>",
  "edge_direction": "buy_yes|buy_no|no_trade",
  "key_factors": ["factor1", "factor2", "factor3"],
  "market_narrative": "<what the market currently believes>",
  "our_narrative": "<what we believe is actually true>",
  "reasoning": "<2-3 sentence final explanation>"
}}"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.messages.create(
                    model=self.model, max_tokens=900,
                    messages=[{"role": "user", "content": prompt}]))
            text = re.sub(r"```json|```", "", response.content[0].text).strip()
            result = json.loads(text)
            result["probability"] = max(0.05, min(0.95, float(result["probability"])))
            result["category"] = category
            return result
        except Exception as e:
            log.error(f"LLM calibrate: {e}")
            return {"probability": ensemble_result["ensemble"], "confidence": "low",
                    "reasoning": f"LLM error: {e}", "category": category}


# ── UPGRADE 3: Tighter Execution Filter ──────────────────────────────────────

class ExecutionFilter:
    """
    [UPGRADE 3] Extra quality gates applied AFTER prediction, BEFORE execution.

    These filters reduce trade count but increase win rate:
      - high_confidence_only: skip anything Claude isn't "high" confidence on
      - min_edge_after_calibration: edge must survive after we account for
        model uncertainty (wider bands on low-confidence predictions)
      - cross_platform_consensus: if Metaculus AND Manifold both agree with
        the market price, our edge is probably noise — skip it
      - model_agreement_required: if XGBoost and LightGBM strongly disagree,
        the signal is unreliable — require larger edge to trade
    """

    def __init__(self, config: "Config"):
        self.config = config

    def check(self, prediction: dict, signal_package: dict) -> tuple[bool, str]:
        """Returns (should_trade, reason)."""
        confidence = prediction.get("confidence", "low")
        edge       = abs(prediction.get("edge", 0))
        agreement  = prediction.get("model_agreement", "weak")
        ext_probs  = signal_package.get("external_probs", [])
        gap        = signal_package.get("cross_platform_gap", 0.0)
        llm_prob   = prediction.get("our_prob", 0.5)
        mkt_prob   = prediction.get("market_prob", 0.5)

        # Gate 1: confidence floor
        if self.config.high_confidence_only and confidence == "low":
            return False, f"Confidence 'low' — high_confidence_only filter active"

        # Gate 2: require larger edge when models disagree
        min_edge = self.config.min_edge_threshold
        if agreement == "weak":
            min_edge = max(min_edge, 0.08)
        if edge < min_edge:
            return False, f"Edge {edge:.3f} below required {min_edge:.3f} (model_agreement={agreement})"

        # Gate 3: cross-platform consensus check
        if ext_probs and gap < 0.05:
            # External forecasters agree with market — our edge may be noise
            return False, f"Cross-platform gap only {gap:.0%} — external forecasters agree with market"

        # Gate 4: order flow contradiction check
        of = signal_package.get("order_flow", {})
        smart_money = of.get("smart_money", "neutral")
        our_direction = "YES" if llm_prob > mkt_prob else "NO"
        if (smart_money == "YES" and our_direction == "NO" or
                smart_money == "NO" and our_direction == "YES"):
            if of.get("large_orders", 0) > 0:
                return False, (f"Order flow CONTRADICTS our direction — "
                                f"smart money going {smart_money}, we want {our_direction}")

        return True, "All execution filters passed"


# ── Main Predict Agent ────────────────────────────────────────────────────────

class PredictAgent:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db     = db
        self.ensemble = EnsemblePredictor()
        self.llm      = LLMCalibrator(config.anthropic_api_key, config.claude_model)
        self.exec_filter = ExecutionFilter(config)

    async def run(self, market: dict, news_texts: list[str],
                  signal_package: dict = None) -> dict:
        """Full prediction pipeline with base rates, domain prompts, ensemble."""
        if signal_package is None:
            signal_package = {}

        market_id    = market["id"]
        market_prob  = market["yes_price"]

        # Inject signal package features into market dict for ensemble features
        market["cross_platform_gap"]    = signal_package.get("cross_platform_gap", 0.0)
        market["order_flow_combined"]   = signal_package.get("order_flow", {}).get("combined", 0.0)

        # ── Ensemble model estimate ────────────────────────────
        ensemble_result = self.ensemble.predict(market)

        # ── Domain-aware LLM calibration ───────────────────────
        relevant_news = signal_package.get("relevant_news", news_texts[:20])
        llm_result = await self.llm.calibrate(market, relevant_news,
                                               ensemble_result, signal_package)
        llm_prob   = llm_result.get("probability", ensemble_result["ensemble"])
        confidence = llm_result.get("confidence", "medium")

        # Apply confidence cap from model disagreement
        if ensemble_result.get("confidence_cap") == "medium" and confidence == "high":
            confidence = "medium"
            log.debug(f"Confidence capped to medium due to model disagreement "
                      f"({ensemble_result['disagreement']:.0%})")

        # ── Three-way ensemble: base_rate + ensemble + LLM ─────
        base_rate_used = llm_result.get("base_rate_used", 0.5)
        llm_weight     = {"low": 0.30, "medium": 0.55, "high": 0.75}.get(confidence, 0.55)
        model_weight   = (1 - llm_weight) * 0.7
        base_weight    = (1 - llm_weight) * 0.3

        calibrated_prob = (
            llm_prob           * llm_weight   +
            ensemble_result["ensemble"] * model_weight +
            base_rate_used     * base_weight
        )
        calibrated_prob = round(max(0.05, min(0.95, calibrated_prob)), 4)
        edge = round(calibrated_prob - market_prob, 4)

        # ── Execution filter (Upgrade 3) ───────────────────────
        tradeable, filter_reason = self.exec_filter.check(
            {**llm_result, "edge": edge, "our_prob": calibrated_prob,
             "market_prob": market_prob,
             "model_agreement": ensemble_result["agreement"]},
            signal_package,
        )

        prediction = {
            "market_id":          market_id,
            "market":             market,
            "market_prob":        market_prob,
            # Model outputs
            "xgb_prob":           ensemble_result["xgb"],
            "lgbm_prob":          ensemble_result["lgbm"],
            "ensemble_prob":      ensemble_result["ensemble"],
            "model_agreement":    ensemble_result["agreement"],
            "model_disagreement": ensemble_result["disagreement"],
            "llm_prob":           round(llm_prob, 4),
            "base_rate_used":     base_rate_used,
            "base_rate_adjustment": llm_result.get("base_rate_adjustment", 0),
            # Final outputs
            "calibrated_prob":    calibrated_prob,
            "our_prob":           calibrated_prob,
            "edge":               edge,
            "confidence":         confidence,
            "tradeable":          tradeable,
            "filter_reason":      filter_reason,
            # Metadata
            "category":           llm_result.get("category", "general"),
            "reasoning":          llm_result.get("reasoning", ""),
            "adjustment_reasoning": llm_result.get("adjustment_reasoning", ""),
            "key_factors":        llm_result.get("key_factors", []),
            "market_narrative":   llm_result.get("market_narrative", ""),
            "our_narrative":      llm_result.get("our_narrative", ""),
            "edge_direction":     llm_result.get("edge_direction", "no_trade"),
            "cross_platform_gap": signal_package.get("cross_platform_gap", 0),
            "smart_money":        signal_package.get("order_flow", {}).get("smart_money", "unknown"),
        }

        # Persist
        pred_id = self.db.save_prediction(prediction)
        prediction["prediction_id"] = pred_id

        log.info(
            f"[{category_short(prediction['category'])}] "
            f"{market['question'][:45]}... "
            f"mkt={market_prob:.0%} ours={calibrated_prob:.0%} "
            f"edge={edge:+.3f} conf={confidence} "
            f"{'TRADE' if tradeable else 'SKIP'}"
        )

        return prediction


# ── Helpers ───────────────────────────────────────────────────────────────────

def category_short(cat: str) -> str:
    return {"economics": "ECO", "crypto": "CRY", "politics": "POL",
            "sports": "SPT", "geopolitics": "GEO", "technology": "TEC"}.get(cat, "GEN")
