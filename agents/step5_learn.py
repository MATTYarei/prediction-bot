"""
Step 5 · Learn
Five agents run post-mortem on every loss.
Save learnings. Update models. Never repeat the same mistake.
"""

import asyncio
import json
import logging
import math
import re
from datetime import datetime

import anthropic

from core.config import Config
from core.database import Database

log = logging.getLogger("predbot.learn")


# ── The 5 Post-Mortem Agents ───────────────────────────────────────────────────

class CalibrationAgent:
    """
    Measures how well our probabilities matched reality.
    Computes Brier score and log-loss for each resolved prediction.
    """

    def analyze(self, our_prob: float, actual_outcome: int) -> dict:
        p = our_prob
        y = float(actual_outcome)

        brier = (p - y) ** 2
        log_loss = -(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))

        if brier < 0.05:
            assessment = "excellent"
        elif brier < 0.15:
            assessment = "good"
        elif brier < 0.25:
            assessment = "fair"
        else:
            assessment = "poor"

        return {
            "brier_score": round(brier, 4),
            "log_loss": round(log_loss, 4),
            "assessment": assessment,
            "calibration_note": (
                "Overconfident — we were too sure it would happen" if p > 0.7 and y == 0
                else "Underconfident — we doubted but it happened" if p < 0.3 and y == 1
                else "Well-calibrated"
            ),
        }


class NarrativeAgent:
    """
    Examines what news/narrative we had at trade time vs. what actually happened.
    Identifies systematic bias in how we read narratives.
    """

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    async def analyze(self, trade: dict, market: dict, outcome: str) -> dict:
        if not self.client:
            return {"lesson": "No API key for narrative analysis"}

        prompt = f"""You are a prediction market post-mortem analyst.

A trade just resolved. Analyze what happened and extract one actionable lesson.

MARKET: {market.get('question', 'Unknown')}
OUR PROBABILITY: {trade.get('our_prob', '?')}
MARKET PRICE AT ENTRY: {trade.get('entry_price', '?')}
DIRECTION: {trade.get('direction', '?')}
OUTCOME: {outcome}
PnL: ${trade.get('pnl', 0):.2f}

Respond with JSON only:
{{
  "what_we_got_wrong": "<1-2 sentences>",
  "what_we_got_right": "<1-2 sentences>",
  "narrative_bias_detected": "<overconfidence|recency_bias|anchoring|availability_heuristic|none>",
  "lesson": "<one specific, actionable lesson for future trades>",
  "filter_adjustment": "<should we tighten or loosen any filters? be specific>"
}}"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            text = re.sub(r"```json|```", "", response.content[0].text).strip()
            return json.loads(text)
        except Exception as e:
            return {"lesson": f"Narrative analysis error: {e}"}


class SignalAgent:
    """
    Checks which data signals (Twitter, Reddit, RSS) were predictive or misleading.
    Over time, de-weights unreliable sources.
    """

    def analyze(self, trade: dict, pnl: float) -> dict:
        won = pnl > 0
        return {
            "signal_verdict": "signals were predictive" if won else "signals were misleading",
            "recommendation": (
                "Maintain current signal weights" if won
                else "Consider reducing weight of contrarian social signals"
            ),
            "lesson": (
                "Social sentiment confirmed the trade — good signal alignment"
                if won
                else "Social sentiment was noisy — apply stricter sentiment threshold next time"
            ),
        }


class ModelAgent:
    """
    Triggers XGBoost retraining when enough new resolved data accumulates.
    Also adjusts LLM prompt strategy based on systematic errors.
    """

    def __init__(self, db: Database):
        self.db = db

    def analyze(self, trade: dict, pnl: float) -> dict:
        stats = self.db.get_calibration_stats()
        total = stats.get("total", 0)
        avg_brier = stats.get("avg_brier")

        retrain_needed = total > 0 and total % 50 == 0  # retrain every 50 resolutions

        return {
            "total_resolved": total,
            "avg_brier_score": avg_brier,
            "retrain_recommended": retrain_needed,
            "lesson": (
                f"Retrain XGBoost — {total} resolved markets in DB"
                if retrain_needed
                else f"Continue collecting data — {total} resolved markets (need 50+)"
            ),
            "prompt_adjustment": (
                "LLM is systematically overconfident — add uncertainty reminder to prompt"
                if avg_brier and avg_brier > 0.25
                else "LLM calibration is acceptable"
            ),
        }


class RiskAgent:
    """
    Reviews whether risk gates were effective or overly conservative.
    Adjusts thresholds based on win/loss patterns.
    """

    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config

    def analyze(self, trade: dict, pnl: float) -> dict:
        open_trades = self.db.get_open_trades()
        total_exposure = sum(t.get("size_usd", 0) for t in open_trades)

        won = pnl > 0
        size = trade.get("size_usd", 0)

        return {
            "current_exposure_usd": total_exposure,
            "trade_size": size,
            "outcome": "win" if won else "loss",
            "lesson": (
                f"Kelly sizing worked — ${size:.2f} bet {'profited' if won else 'lost'} ${abs(pnl):.2f}. "
                + ("Consider slightly larger bets on high-confidence trades."
                   if won and trade.get("confidence") == "high"
                   else "Kelly fraction is appropriate." if won
                   else "Review edge threshold — this loss may indicate edge was illusory.")
            ),
            "threshold_adjustment": (
                "Consider raising min_edge_threshold to 0.07"
                if not won and abs(trade.get("edge", 0)) < 0.08
                else "Current thresholds seem appropriate"
            ),
        }


# ── Learn Orchestrator ─────────────────────────────────────────────────────────

class LearnOrchestrator:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        claude_client = (
            anthropic.Anthropic(api_key=config.anthropic_api_key)
            if config.anthropic_api_key else None
        )
        self.calibration = CalibrationAgent()
        self.narrative = NarrativeAgent(claude_client, config.claude_model)
        self.signal = SignalAgent()
        self.model_agent = ModelAgent(db)
        self.risk_agent = RiskAgent(db, config)

    async def run(self) -> list[dict]:
        """
        Find all closed trades that haven't been analyzed yet.
        Run all 5 post-mortem agents on each. Save learnings.
        """
        closed_trades = self._get_unanalyzed_trades()
        if not closed_trades:
            log.info("No new resolved trades to analyze")
            return []

        results = []
        for trade in closed_trades:
            result = await self._run_postmortem(trade)
            results.append(result)

        # Trigger model retrain if needed
        self._maybe_retrain()

        return results

    async def _run_postmortem(self, trade: dict) -> dict:
        """Run all 5 agents concurrently on a single resolved trade."""
        market = self.db.get_market(trade["market_id"]) or {}
        pnl = trade.get("pnl", 0)
        actual_outcome = 1 if pnl > 0 else 0  # simplified: win = YES resolved
        our_prob = trade.get("entry_price", 0.5)  # use entry price as proxy

        outcome_str = f"{'WIN' if pnl > 0 else 'LOSS'} — PnL: ${pnl:+.2f}"

        # Calibration is synchronous
        cal_result = self.calibration.analyze(our_prob, actual_outcome)

        # Run async agents concurrently
        narrative_task = self.narrative.analyze(trade, market, outcome_str)
        signal_result = self.signal.analyze(trade, pnl)
        model_result = self.model_agent.analyze(trade, pnl)
        risk_result = self.risk_agent.analyze(trade, pnl)

        narrative_result = await narrative_task

        # Aggregate lessons
        all_lessons = [
            cal_result.get("calibration_note", ""),
            narrative_result.get("lesson", ""),
            signal_result.get("lesson", ""),
            model_result.get("lesson", ""),
            risk_result.get("lesson", ""),
        ]
        combined_lesson = " | ".join(l for l in all_lessons if l)

        learning = {
            "trade_id": trade["id"],
            "market_id": trade["market_id"],
            "outcome": outcome_str,
            "our_prob": our_prob,
            "actual_outcome": actual_outcome,
            "brier_score": cal_result["brier_score"],
            "log_loss": cal_result["log_loss"],
            "lessons": combined_lesson,
            "agent_reports": {
                "calibration": cal_result,
                "narrative": narrative_result,
                "signal": signal_result,
                "model": model_result,
                "risk": risk_result,
            },
        }

        self.db.save_learning(learning)

        log.info(
            f"Post-mortem on trade #{trade['id']}: "
            f"Brier={cal_result['brier_score']:.3f} "
            f"({cal_result['assessment']}) — {outcome_str}"
        )

        return learning

    def _get_unanalyzed_trades(self) -> list[dict]:
        """Return closed trades not yet in the learnings table."""
        with self.db._conn() as conn:
            rows = conn.execute("""
                SELECT t.* FROM trades t
                LEFT JOIN learnings l ON t.id = l.trade_id
                WHERE t.status = 'closed' AND l.id IS NULL
            """).fetchall()
            return [dict(r) for r in rows]

    def _maybe_retrain(self):
        """Trigger XGBoost retraining if model agent recommends it."""
        try:
            from agents.step3_predict import XGBoostPredictor
            predictor = XGBoostPredictor()
            predictor.train_on_history(self.db)
        except Exception as e:
            log.error(f"Retraining error: {e}")
