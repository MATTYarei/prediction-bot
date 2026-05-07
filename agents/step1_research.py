"""
Step 1 · Research  [v3 — Kalshi primary, Prediction Hunt, arb scanning]

Changes from v2:
  - KalshiFetcher:        fetches markets from Kalshi (primary US venue)
  - PredictionHuntClient: unified API covering Kalshi+Polymarket+PredictIt+Opinion
  - ArbScanner:           finds price gaps between Kalshi and Polymarket
    on the same underlying event — flags pure arbitrage opportunities
  - ResearchOrchestrator: now returns arb_opportunities alongside signal packages
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Optional

import anthropic

from core.config import Config
from core.database import Database

log = logging.getLogger("predbot.research")


# ── Kalshi Market Fetcher (NEW primary) ───────────────────────────────────────

class KalshiFetcher:
    """
    Fetches active markets from Kalshi's REST API.

    Kalshi market format differs from Polymarket:
      - Markets identified by ticker strings (e.g. "INXD-23DEC31-B5000")
      - Prices expressed as yes_bid/yes_ask (use midpoint)
      - Categories: politics, economics, sports, entertainment, climate, etc.
      - No crypto wallet required — fiat USD

    Kalshi is the recommended primary source for US-based traders.
    CFTC-regulated, customer funds segregated, 900+ active markets.
    """

    def __init__(self, base_url: str, api_key_id: str = "",
                 private_key_path: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self._private_key = None
        self._load_key()

    def _load_key(self):
        if not self.private_key_path:
            return
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            with open(self.private_key_path, "rb") as f:
                self._private_key = load_pem_private_key(f.read(), password=None)
        except Exception as e:
            log.debug(f"Kalshi key load: {e}")

    def _sign(self, method: str, path: str) -> dict:
        if not self._private_key or not self.api_key_id:
            return {}
        try:
            import time, base64
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            ts  = str(int(time.time() * 1000))
            msg = f"{ts}{method.upper()}{path}".encode()
            sig = self._private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
            return {
                "KALSHI-ACCESS-KEY":       self.api_key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            }
        except Exception as e:
            log.debug(f"Kalshi sign: {e}")
            return {}

    async def fetch_markets(self, limit: int = 200) -> list[dict]:
        """Fetch open Kalshi markets and normalise to the bot's standard format."""
        try:
            import httpx
            path = "/markets"
            headers = self._sign("GET", path)
            params  = {"limit": limit, "status": "open"}
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(f"{self.base_url}{path}",
                                     headers=headers, params=params)
                r.raise_for_status()
                raw_markets = r.json().get("markets", [])
                result = []
                for m in raw_markets:
                    yes_bid = float(m.get("yes_bid", 0.5))
                    yes_ask = float(m.get("yes_ask", 0.5))
                    yes_price = (yes_bid + yes_ask) / 2
                    # Kalshi uses cents internally; prices should already be 0-1
                    if yes_price > 1:
                        yes_price /= 100
                    result.append({
                        "id":            m.get("ticker", ""),
                        "kalshi_ticker": m.get("ticker", ""),
                        "platform":      "kalshi",
                        "question":      m.get("title", m.get("question", "")),
                        "yes_price":     round(yes_price, 4),
                        "yes_bid":       yes_bid if yes_bid <= 1 else yes_bid / 100,
                        "yes_ask":       yes_ask if yes_ask <= 1 else yes_ask / 100,
                        "volume_usd":    float(m.get("volume", 0)),
                        "liquidity_usd": float(m.get("open_interest", 0)),
                        "end_date":      m.get("close_time", ""),
                        "category":      m.get("category", "").lower(),
                        "subtitle":      m.get("subtitle", ""),
                    })
                log.info(f"Kalshi: fetched {len(result)} markets")
                return result
        except Exception as e:
            log.error(f"Kalshi fetch_markets: {e}")
            return _mock_markets("kalshi")


# ── Prediction Hunt Client (NEW unified API) ──────────────────────────────────

class PredictionHuntClient:
    """
    Prediction Hunt is a unified API covering 5 platforms:
      Kalshi, Polymarket, PredictIt, ProphetX, Opinion

    One API key → data from all platforms → enables cross-platform comparison.

    Key endpoints we use:
      /markets          list markets across all platforms
      /arb              pre-computed arbitrage opportunities
      /market/{id}      single market with all platform prices

    Get your API key at: predictionhunt.com
    """

    def __init__(self, api_key: str,
                 base_url: str = "https://www.predictionhunt.com/api/v2"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    async def get_markets(self, platforms: list[str] = None,
                          limit: int = 200) -> list[dict]:
        """
        Fetch markets across all platforms via Prediction Hunt's unified API.
        Returns normalised market dicts compatible with the rest of the bot.
        """
        if not self.api_key:
            log.debug("Prediction Hunt: no API key — skipping")
            return []
        try:
            import httpx
            params = {"limit": limit}
            if platforms:
                params["platforms"] = ",".join(platforms)
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(f"{self.base_url}/markets",
                                     headers=self._headers, params=params)
                r.raise_for_status()
                raw = r.json().get("markets", [])
                result = []
                for m in raw:
                    prices = m.get("platform_prices", {})
                    yes_price = float(m.get("best_price", m.get("yes_price", 0.5)))
                    result.append({
                        "id":               m.get("id", ""),
                        "platform":         m.get("platform", "prediction_hunt"),
                        "question":         m.get("question", m.get("title", "")),
                        "yes_price":        round(yes_price, 4),
                        "volume_usd":       float(m.get("volume_usd", 0)),
                        "liquidity_usd":    float(m.get("liquidity_usd", 0)),
                        "end_date":         m.get("close_time", m.get("end_date", "")),
                        "category":         m.get("category", "").lower(),
                        "platform_prices":  prices,
                        "kalshi_ticker":    m.get("kalshi_ticker", ""),
                        "polymarket_id":    m.get("polymarket_id", ""),
                    })
                log.info(f"Prediction Hunt: {len(result)} cross-platform markets")
                return result
        except Exception as e:
            log.error(f"Prediction Hunt get_markets: {e}")
            return []

    async def get_arb_opportunities(self, min_gap: float = 0.04) -> list[dict]:
        """
        Fetch pre-computed arbitrage opportunities from Prediction Hunt.
        These are markets where the same event has meaningfully different
        prices on Kalshi vs Polymarket (or other platforms).

        min_gap: minimum price difference to return (default 4%)
        """
        if not self.api_key:
            return []
        try:
            import httpx
            params = {"min_gap": min_gap, "platforms": "kalshi,polymarket"}
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{self.base_url}/arb",
                                     headers=self._headers, params=params)
                r.raise_for_status()
                arbs = r.json().get("opportunities", [])
                result = []
                for a in arbs:
                    result.append({
                        "question":          a.get("question", ""),
                        "kalshi_ticker":     a.get("kalshi_ticker", ""),
                        "polymarket_id":     a.get("polymarket_id", ""),
                        "kalshi_price":      float(a.get("kalshi_price", 0.5)),
                        "polymarket_price":  float(a.get("polymarket_price", 0.5)),
                        "gap":               float(a.get("gap", 0)),
                        "direction":         a.get("direction", ""),
                        "est_profit_pct":    float(a.get("est_profit_pct", 0)),
                        "source":            "prediction_hunt",
                    })
                log.info(f"Prediction Hunt: {len(result)} arb opportunities "
                         f"(min_gap={min_gap:.0%})")
                return result
        except Exception as e:
            log.error(f"Prediction Hunt arb: {e}")
            return []


# ── Arbitrage Scanner (local, no Prediction Hunt key needed) ─────────────────

class ArbScanner:
    """
    Finds price gaps between Kalshi and Polymarket markets by matching
    questions using fuzzy string similarity.

    This runs locally when Prediction Hunt isn't configured, or in addition
    to Prediction Hunt to catch gaps it might miss.

    A gap of ≥4% on a binary contract is exploitable because:
      Buy YES on the cheaper platform + Buy NO on the expensive platform
      = guaranteed profit of (gap - fees) regardless of outcome.
    """

    def __init__(self, min_gap: float = 0.04):
        self.min_gap = min_gap

    def find_opportunities(self, kalshi_markets: list[dict],
                           poly_markets: list[dict]) -> list[dict]:
        """Match markets across platforms and return price gaps above threshold."""
        opportunities = []

        for k in kalshi_markets:
            k_q = _normalise_question(k["question"])
            best_match = None
            best_score = 0.0

            for p in poly_markets:
                p_q = _normalise_question(p["question"])
                score = _similarity(k_q, p_q)
                if score > best_score:
                    best_score = score
                    best_match = p

            if best_match is None or best_score < 0.55:
                continue

            k_price = k["yes_price"]
            p_price = best_match["yes_price"]
            gap     = k_price - p_price  # positive = Kalshi higher

            if abs(gap) < self.min_gap:
                continue

            direction = ("buy_poly_sell_kalshi" if gap > 0
                         else "buy_kalshi_sell_poly")

            opportunities.append({
                "question":         k["question"],
                "kalshi_ticker":    k.get("kalshi_ticker", k["id"]),
                "polymarket_id":    best_match["id"],
                "kalshi_price":     k_price,
                "polymarket_price": p_price,
                "gap":              round(abs(gap), 4),
                "direction":        direction,
                "match_score":      round(best_score, 3),
                "est_profit_pct":   round(abs(gap) * 0.9, 4),  # rough net of fees
                "source":           "local_scan",
            })

        opportunities.sort(key=lambda x: x["gap"], reverse=True)
        log.info(f"ArbScanner: {len(opportunities)} opportunities "
                 f"(min_gap={self.min_gap:.0%})")
        return opportunities


# ── Existing scraper agents (from v2, unchanged) ──────────────────────────────

class TwitterAgent:
    def __init__(self, bearer_token: str):
        self.token = bearer_token
        self.base = "https://api.twitter.com/2"

    async def fetch(self, query: str, max_results: int = 50) -> list[dict]:
        if not self.token:
            return []
        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.token}"}
            params  = {"query": f"{query} -is:retweet lang:en",
                       "max_results": min(max_results, 100),
                       "tweet.fields": "created_at,public_metrics"}
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{self.base}/tweets/search/recent",
                                     headers=headers, params=params)
                r.raise_for_status()
                tweets = r.json().get("data", [])
                return [{"source": "twitter", "text": t["text"],
                         "weight": 1.0 + min(t.get("public_metrics", {}).get(
                             "like_count", 0) / 100, 2.0)} for t in tweets]
        except Exception as e:
            log.error(f"Twitter: {e}")
            return []

    async def fetch_for_markets(self, markets: list[dict]) -> list[dict]:
        tasks   = [self.fetch(_market_to_query(m["question"]), 20)
                   for m in markets[:10]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [item for r in results if isinstance(r, list) for item in r]


class RedditAgent:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self._token: Optional[str] = None

    async def _get_token(self) -> Optional[str]:
        if not self.client_id:
            return None
        try:
            import httpx, base64
            creds = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()).decode()
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    "https://www.reddit.com/api/v1/access_token",
                    headers={"Authorization": f"Basic {creds}",
                             "User-Agent": self.user_agent},
                    data={"grant_type": "client_credentials"})
                r.raise_for_status()
                return r.json().get("access_token")
        except Exception as e:
            log.error(f"Reddit auth: {e}")
            return None

    async def fetch_subreddit(self, subreddit: str, limit: int = 25) -> list[dict]:
        token = self._token or await self._get_token()
        if not token:
            return []
        self._token = token
        try:
            import httpx
            headers = {"Authorization": f"Bearer {token}",
                       "User-Agent": self.user_agent}
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"https://oauth.reddit.com/r/{subreddit}/hot",
                    headers=headers, params={"limit": limit})
                r.raise_for_status()
                posts = r.json().get("data", {}).get("children", [])
                return [{"source": f"reddit/r/{subreddit}",
                         "text": f"{p['data']['title']} {p['data'].get('selftext','')[:300]}",
                         "weight": 1.0 + min(p["data"].get("score", 0) / 500, 2.0)}
                        for p in posts]
        except Exception as e:
            log.error(f"Reddit r/{subreddit}: {e}")
            return []

    async def fetch_all(self, subreddits: list[str]) -> list[dict]:
        results = await asyncio.gather(
            *[self.fetch_subreddit(s) for s in subreddits],
            return_exceptions=True)
        flat = [item for r in results if isinstance(r, list) for item in r]
        log.info(f"Reddit: {len(flat)} posts")
        return flat


class RSSAgent:
    def __init__(self, feeds: list[str]):
        self.feeds = feeds

    async def fetch_all(self) -> list[dict]:
        loop    = asyncio.get_event_loop()
        results = await asyncio.gather(
            *[loop.run_in_executor(None, self._parse_feed, url)
              for url in self.feeds],
            return_exceptions=True)
        articles = [item for r in results if isinstance(r, list) for item in r]
        log.info(f"RSS: {len(articles)} articles")
        return articles

    def _parse_feed(self, url: str) -> list[dict]:
        try:
            import feedparser
            feed = feedparser.parse(url)
            return [{"source": feed.feed.get("title", url),
                     "headline": e.get("title", ""),
                     "summary": re.sub(r"<[^>]+>", "", e.get("summary", ""))[:500],
                     "url": e.get("link", ""),
                     "text": e.get("title", "") + " " + e.get("summary", ""),
                     "weight": 1.2} for e in feed.entries[:20]]
        except Exception as e:
            log.error(f"RSS {url}: {e}")
            return []


class OrderFlowAgent:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base = base_url
        self.api_key = api_key

    async def fetch_order_book(self, token_id: str) -> dict:
        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base}/book",
                                     headers=headers, params={"token_id": token_id})
                r.raise_for_status()
                return r.json()
        except Exception:
            return {}

    async def fetch_recent_trades(self, token_id: str, limit: int = 50) -> list[dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base}/trades",
                                     params={"token_id": token_id, "limit": limit})
                r.raise_for_status()
                return r.json().get("data", [])
        except Exception:
            return []

    def analyze_book(self, book: dict, trades: list[dict]) -> dict:
        if not book:
            return {"imbalance": 0.0, "signal": "none", "large_orders": 0,
                    "smart_money_direction": "unknown", "combined_signal": 0.0,
                    "weight": 0.5}
        bids     = book.get("bids", [])
        asks     = book.get("asks", [])
        bid_vol  = sum(float(b.get("size", 0)) for b in bids[:10])
        ask_vol  = sum(float(a.get("size", 0)) for a in asks[:10])
        total    = bid_vol + ask_vol
        imbal    = (bid_vol - ask_vol) / total if total > 0 else 0.0
        large    = len([b for b in bids if float(b.get("size", 0)) > 500]) + \
                   len([a for a in asks if float(a.get("size", 0)) > 500])
        yes_b    = sum(float(t.get("size", 0)) for t in trades
                       if t.get("side", "").upper() == "BUY")
        no_b     = sum(float(t.get("size", 0)) for t in trades
                       if t.get("side", "").upper() == "SELL")
        flow     = (yes_b - no_b) / (yes_b + no_b + 1e-9)
        combined = imbal * 0.6 + flow * 0.4
        signal   = ("strongly_bullish" if combined > 0.3 else
                    "bullish"          if combined > 0.1 else
                    "bearish"          if combined < -0.1 else
                    "strongly_bearish" if combined < -0.3 else "neutral")
        return {"imbalance": round(imbal, 3), "trade_flow": round(flow, 3),
                "combined_signal": round(combined, 3), "signal": signal,
                "large_orders": large,
                "smart_money_direction": ("YES" if combined > 0.05
                                          else "NO" if combined < -0.05
                                          else "neutral"),
                "weight": 2.0 if large > 0 else 1.5}

    async def analyze_market(self, market: dict) -> dict:
        tid = market.get("id", "")
        if not tid:
            return {}
        book, trades = await asyncio.gather(
            self.fetch_order_book(tid), self.fetch_recent_trades(tid),
            return_exceptions=True)
        book   = book   if isinstance(book, dict)   else {}
        trades = trades if isinstance(trades, list) else []
        r = self.analyze_book(book, trades)
        r["market_id"] = tid
        return r

    async def analyze_markets(self, markets: list[dict]) -> dict:
        top = sorted(markets, key=lambda m: m.get("volume_usd", 0),
                     reverse=True)[:20]
        results = await asyncio.gather(
            *[self.analyze_market(m) for m in top], return_exceptions=True)
        return {r["market_id"]: r for r in results
                if isinstance(r, dict) and r.get("market_id")}


class MetaculusAgent:
    BASE = "https://www.metaculus.com/api2"

    async def search_question(self, query: str) -> list[dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{self.BASE}/questions/",
                    params={"search": query[:100], "status": "open",
                            "forecast_type": "binary", "order_by": "-activity",
                            "limit": 5},
                    headers={"Accept": "application/json"})
                r.raise_for_status()
                results = []
                for q in r.json().get("results", []):
                    prob = q.get("community_prediction", {}).get("full", {}).get("q2")
                    if prob is not None:
                        results.append({
                            "source": "metaculus",
                            "question": q.get("title", ""),
                            "metaculus_prob": float(prob),
                            "forecaster_count": q.get("number_of_forecasters", 0),
                            "weight": min(3.0, 1.0 + q.get("number_of_forecasters",
                                                             0) / 100),
                        })
                return results
        except Exception as e:
            log.debug(f"Metaculus: {e}")
            return []

    async def fetch_for_markets(self, markets: list[dict]) -> dict:
        top = sorted(markets, key=lambda m: m.get("volume_usd", 0),
                     reverse=True)[:15]
        fetched = await asyncio.gather(
            *[self.search_question(_market_to_query(m["question"])) for m in top],
            return_exceptions=True)
        return {m["id"]: r for m, r in zip(top, fetched)
                if isinstance(r, list) and r}


class ManifoldAgent:
    BASE = "https://api.manifold.markets/v0"

    async def search_markets(self, query: str) -> list[dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{self.BASE}/search-markets",
                    params={"term": query[:100], "limit": 5,
                            "filter": "open", "sort": "liquidity"})
                r.raise_for_status()
                return [{"source": "manifold", "question": m.get("question", ""),
                         "manifold_prob": float(m["probability"]), "weight": 0.7}
                        for m in r.json()
                        if m.get("outcomeType") == "BINARY"
                        and m.get("probability") is not None]
        except Exception as e:
            log.debug(f"Manifold: {e}")
            return []

    async def fetch_for_markets(self, markets: list[dict]) -> dict:
        top = sorted(markets, key=lambda m: m.get("volume_usd", 0),
                     reverse=True)[:15]
        fetched = await asyncio.gather(
            *[self.search_markets(_market_to_query(m["question"])) for m in top],
            return_exceptions=True)
        return {m["id"]: r for m, r in zip(top, fetched)
                if isinstance(r, list) and r}


class SignalAggregator:
    def build_market_signal_package(self, market: dict, order_flow: dict,
                                    metaculus_matches: list[dict],
                                    manifold_matches: list[dict],
                                    all_news: list[dict]) -> dict:
        question = market["question"]
        external_probs = []
        for m in (metaculus_matches or [])[:2]:
            external_probs.append({"source": "metaculus", "prob": m["metaculus_prob"],
                                    "weight": m["weight"],
                                    "forecasters": m.get("forecaster_count", 0)})
        for m in (manifold_matches or [])[:1]:
            external_probs.append({"source": "manifold",
                                    "prob": m["manifold_prob"], "weight": m["weight"]})
        if external_probs:
            total_w = sum(e["weight"] for e in external_probs)
            consensus_prob = sum(e["prob"] * e["weight"]
                                  for e in external_probs) / total_w
            cross_platform_gap = abs(consensus_prob - market["yes_price"])
        else:
            consensus_prob, cross_platform_gap = None, 0.0

        of = order_flow or {}
        of_combined = of.get("combined_signal", 0.0)
        keywords = set(re.sub(r"[^a-z ]", "", question.lower()).split()) - \
                   {"will", "the", "a", "an", "be", "in", "of", "to",
                    "is", "are", "was", "for", "and", "or", "by"}
        relevant_texts = sorted(
            [{"text": item["text"],
              "weight": item.get("weight", 1.0) * sum(1 for k in keywords
                                                        if k in item.get("text","").lower()),
              "source": item.get("source", "")}
             for item in all_news
             if sum(1 for k in keywords if k in item.get("text","").lower()) >= 2],
            key=lambda x: x["weight"], reverse=True)

        signal_strength = (
            min(cross_platform_gap * 5, 1.0) * 0.40 +
            min(abs(of_combined) * 2, 1.0)   * 0.35 +
            min(len(relevant_texts) / 10, 1.0) * 0.25
        )
        return {
            "market_id":          market["id"],
            "market_price":       market["yes_price"],
            "external_probs":     external_probs,
            "consensus_prob":     round(consensus_prob, 4) if consensus_prob else None,
            "cross_platform_gap": round(cross_platform_gap, 3),
            "order_flow":         {"signal": of.get("signal", "none"),
                                    "combined": of_combined,
                                    "smart_money": of.get("smart_money_direction",
                                                           "unknown"),
                                    "large_orders": of.get("large_orders", 0)},
            "relevant_news":      [t["text"] for t in relevant_texts[:15]],
            "signal_strength":    round(signal_strength, 3),
            "worth_trading":      signal_strength > 0.2,
        }


class SentimentAgent:
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.model = model

    async def analyze(self, texts: list[str], market_question: str,
                      external_probs: list[dict] = None) -> dict:
        if not self.client or not texts:
            return {"score": 0.0, "label": "neutral", "summary": "No data"}
        sample    = "\n".join(f"- {t[:250]}" for t in texts[:25])
        ext_block = ""
        if external_probs:
            ext_block = "\nEXTERNAL FORECASTS:\n" + "\n".join(
                f"- {e['source'].title()}: {e['prob']:.0%} YES"
                + (f" ({e['forecasters']} forecasters)"
                   if e.get("forecasters") else "")
                for e in external_probs)
        prompt = (f"Analyze prediction market signals.\n"
                  f"QUESTION: {market_question}\n{ext_block}\n"
                  f"SIGNALS:\n{sample}\n\n"
                  f'Return JSON: {{"score":<-1.0 to 1.0>,"label":"bearish|neutral|bullish",'
                  f'"key_themes":["t1","t2"],"summary":"<2 sentences>"}}')
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, lambda: self.client.messages.create(
                    model=self.model, max_tokens=400,
                    messages=[{"role": "user", "content": prompt}]))
            text = re.sub(r"```json|```", "", resp.content[0].text).strip()
            return json.loads(text)
        except Exception as e:
            log.error(f"Sentiment: {e}")
            return {"score": 0.0, "label": "neutral", "summary": str(e)}


# ── Polymarket Fetcher (kept as secondary) ────────────────────────────────────

class PolymarketFetcher:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base = base_url.rstrip("/")
        self.api_key = api_key

    async def fetch_markets(self, limit: int = 200) -> list[dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(f"{self.base}/markets",
                                     params={"limit": limit, "active": "true",
                                             "closed": "false"})
                r.raise_for_status()
                raw  = r.json()
                mkts = raw if isinstance(raw, list) else raw.get("data", [])
                result = []
                for m in mkts:
                    tokens    = m.get("tokens", [])
                    yes_price = next(
                        (float(t.get("price", 0.5)) for t in tokens
                         if t.get("outcome", "").upper() == "YES"), 0.5)
                    result.append({
                        "id":            m.get("condition_id", m.get("id", "")),
                        "polymarket_id": m.get("condition_id", m.get("id", "")),
                        "platform":      "polymarket",
                        "question":      m.get("question", ""),
                        "yes_price":     yes_price,
                        "volume_usd":    float(m.get("volume", 0)),
                        "liquidity_usd": float(m.get("liquidity", 0)),
                        "end_date":      m.get("end_date_iso", ""),
                        "category":      m.get("category", "").lower(),
                    })
                log.info(f"Polymarket: {len(result)} markets")
                return result
        except Exception as e:
            log.error(f"Polymarket: {e}")
            return _mock_markets("polymarket")


# ── Research Orchestrator ─────────────────────────────────────────────────────

class ResearchOrchestrator:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db

        # Primary market source depends on venue setting
        self.kalshi_fetcher   = KalshiFetcher(
            config.kalshi_base_url,
            config.kalshi_api_key_id,
            config.kalshi_private_key_path,
        )
        self.polymarket       = PolymarketFetcher(config.polymarket_base_url,
                                                    config.polymarket_api_key)
        self.prediction_hunt  = PredictionHuntClient(
            config.prediction_hunt_api_key)
        self.arb_scanner      = ArbScanner(config.arb_min_gap)
        self.twitter          = TwitterAgent(config.twitter_bearer_token)
        self.reddit           = RedditAgent(config.reddit_client_id,
                                             config.reddit_client_secret,
                                             config.reddit_user_agent)
        self.rss              = RSSAgent(config.rss_feeds)
        self.order_flow       = OrderFlowAgent(config.polymarket_base_url,
                                                config.polymarket_api_key)
        self.metaculus        = MetaculusAgent()
        self.manifold         = ManifoldAgent()
        self.aggregator       = SignalAggregator()
        self.sentiment        = SentimentAgent(config.anthropic_api_key,
                                                config.claude_model)

    async def run(self) -> dict:
        # ── Tier 1: fetch all market sources in parallel ───────
        venue   = self.config.primary_venue
        limit   = self.config.max_markets_to_scan

        if self.config.prediction_hunt_api_key:
            # Prediction Hunt gives us cross-platform data in one call
            (ph_markets, kalshi_markets, poly_markets,
             reddit_posts, rss_articles, ph_arbs) = await asyncio.gather(
                self.prediction_hunt.get_markets(limit=limit),
                self.kalshi_fetcher.fetch_markets(limit // 2),
                self.polymarket.fetch_markets(limit // 2),
                self.reddit.fetch_all(self.config.research_subreddits),
                self.rss.fetch_all(),
                self.prediction_hunt.get_arb_opportunities(self.config.arb_min_gap),
            )
            # Use PH markets as primary (they're already cross-platform)
            primary_markets = ph_markets or kalshi_markets
        else:
            # No PH key — fetch Kalshi + Polymarket directly
            (kalshi_markets, poly_markets,
             reddit_posts, rss_articles) = await asyncio.gather(
                self.kalshi_fetcher.fetch_markets(limit),
                self.polymarket.fetch_markets(limit // 2),
                self.reddit.fetch_all(self.config.research_subreddits),
                self.rss.fetch_all(),
            )
            primary_markets = kalshi_markets
            ph_arbs = []

        # Use Kalshi as the execution-ready market list
        exec_markets = kalshi_markets if venue == "kalshi" else poly_markets
        if not exec_markets:
            exec_markets = primary_markets

        for m in exec_markets:
            self.db.upsert_market(m)

        # ── Tier 2: enrichment signals ─────────────────────────
        twitter_posts, order_flows, metaculus_data, manifold_data = await asyncio.gather(
            self.twitter.fetch_for_markets(exec_markets[:10]),
            self.order_flow.analyze_markets(poly_markets or exec_markets),
            self.metaculus.fetch_for_markets(exec_markets),
            self.manifold.fetch_for_markets(exec_markets),
        )
        all_news = rss_articles + reddit_posts + twitter_posts
        self.db.save_news([{"source": a.get("source", ""),
                             "headline": a.get("headline", ""),
                             "summary": a.get("summary", ""),
                             "sentiment": 0.0, "url": a.get("url", "")}
                            for a in rss_articles])

        # ── Tier 3: local arb scan + merge with PH arbs ────────
        local_arbs = []
        if self.config.arb_scan_enabled and kalshi_markets and poly_markets:
            local_arbs = self.arb_scanner.find_opportunities(
                kalshi_markets, poly_markets)

        # Deduplicate arbs by kalshi_ticker
        all_arbs = {a["kalshi_ticker"]: a for a in ph_arbs + local_arbs}.values()
        arb_opportunities = sorted(all_arbs, key=lambda x: x["gap"], reverse=True)
        if arb_opportunities:
            log.info(f"Arb: {len(arb_opportunities)} opportunities "
                     f"(best gap {arb_opportunities[0]['gap']:.2%})")

        # ── Tier 4: per-market signal packages ─────────────────
        signal_packages = {}
        for m in exec_markets:
            mid = m["id"]
            signal_packages[mid] = self.aggregator.build_market_signal_package(
                market=m,
                order_flow=order_flows.get(mid, {}),
                metaculus_matches=metaculus_data.get(mid, []),
                manifold_matches=manifold_data.get(mid, []),
                all_news=all_news,
            )

        # Global sentiment
        top_market = max(exec_markets,
                          key=lambda m: m.get("volume_usd", 0)) if exec_markets else {}
        top_pkg    = signal_packages.get(top_market.get("id", ""), {})
        global_sentiment = await self.sentiment.analyze(
            texts=[n.get("text", "") for n in all_news[:30]],
            market_question=top_market.get("question", "global conditions"),
            external_probs=top_pkg.get("external_probs", []),
        )

        log.info(f"Research: {len(exec_markets)} {venue} markets | "
                 f"{len(order_flows)} order books | "
                 f"{len(metaculus_data)} metaculus | "
                 f"{len(arb_opportunities)} arbs")

        return {
            "markets":           exec_markets,
            "kalshi_markets":    kalshi_markets,
            "poly_markets":      poly_markets,
            "news":              [n.get("text", "") for n in all_news],
            "news_structured":   all_news,
            "signal_packages":   signal_packages,
            "order_flows":       order_flows,
            "metaculus_data":    metaculus_data,
            "manifold_data":     manifold_data,
            "arb_opportunities": list(arb_opportunities),
            "sentiment":         global_sentiment,
            "sentiment_score":   global_sentiment.get("score", 0.0),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _market_to_query(question: str) -> str:
    return " ".join(re.sub(r"[?!.,]", "", question).split()[:8])


def _normalise_question(q: str) -> str:
    """Lowercase, strip punctuation, remove filler words for matching."""
    q = re.sub(r"[^a-z0-9 ]", "", q.lower())
    stop = {"will", "the", "a", "an", "be", "in", "of", "to", "is", "are",
            "was", "for", "and", "or", "by", "on", "at", "before", "after",
            "during", "this", "that", "with"}
    return " ".join(w for w in q.split() if w not in stop)


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity between two normalised question strings."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _mock_markets(platform: str = "kalshi") -> list[dict]:
    base = datetime.utcnow()
    return [
        {"id": "KXFED-25JUN-Y25" if platform == "kalshi" else "mock-001",
         "kalshi_ticker": "KXFED-25JUN-Y25",
         "platform": platform,
         "question": "Will the Fed cut rates by 25bps in June 2026?",
         "yes_price": 0.62, "volume_usd": 125000, "liquidity_usd": 45000,
         "end_date": (base + timedelta(days=30)).isoformat(), "category": "economics"},
        {"id": "KXBTC-120K-JUL" if platform == "kalshi" else "mock-002",
         "kalshi_ticker": "KXBTC-120K-JUL",
         "platform": platform,
         "question": "Will Bitcoin exceed $120,000 before July 2026?",
         "yes_price": 0.38, "volume_usd": 340000, "liquidity_usd": 120000,
         "end_date": (base + timedelta(days=45)).isoformat(), "category": "crypto"},
        {"id": "KXSPX-6000-DEC" if platform == "kalshi" else "mock-003",
         "kalshi_ticker": "KXSPX-6000-DEC",
         "platform": platform,
         "question": "Will the S&P 500 end 2026 above 6000?",
         "yes_price": 0.71, "volume_usd": 89000, "liquidity_usd": 32000,
         "end_date": (base + timedelta(days=240)).isoformat(), "category": "economics"},
        {"id": "KXREC-2026" if platform == "kalshi" else "mock-004",
         "kalshi_ticker": "KXREC-2026",
         "platform": platform,
         "question": "Will there be a US recession declared in 2026?",
         "yes_price": 0.22, "volume_usd": 210000, "liquidity_usd": 78000,
         "end_date": (base + timedelta(days=180)).isoformat(), "category": "economics"},
    ]
