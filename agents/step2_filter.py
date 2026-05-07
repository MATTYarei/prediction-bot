"""
Step 2 · Filter
Scan 300+ markets. Filter by liquidity, volume, and time to resolution.
Flag the ones worth pursuing.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from core.config import Config
from core.database import Database

log = logging.getLogger("predbot.filter")


class FilterAgent:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db

    async def run(self, markets: list[dict]) -> list[dict]:
        """Score and filter all markets. Return only tradeable candidates."""
        log.info(f"Scanning {len(markets)} markets...")

        scored = []
        stats = {
            "total": len(markets),
            "passed_liquidity": 0,
            "passed_volume": 0,
            "passed_time": 0,
            "passed_price": 0,
            "passed_category": 0,
            "flagged": 0,
        }

        for market in markets:
            result = self._evaluate(market, stats)
            if result:
                scored.append(result)

        # Sort by expected value score descending
        scored.sort(key=lambda x: x["filter_score"], reverse=True)

        stats["flagged"] = len(scored)
        self._log_stats(stats)

        return scored

    def _evaluate(self, market: dict, stats: dict) -> Optional[dict]:
        """
        Apply all filters. Return enriched market dict if it passes, else None.
        """
        q          = market.get("question", "")
        yes_price  = market.get("yes_price", 0.5)
        liquidity  = market.get("liquidity_usd", 0)
        volume     = market.get("volume_usd", 0)
        end_date_str = market.get("end_date", "")
        category   = market.get("category", "").lower()

        if liquidity < self.config.min_liquidity_usd:
            return None
        stats["passed_liquidity"] += 1

        if volume < self.config.min_daily_volume_usd:
            return None
        stats["passed_volume"] += 1

        days_left = self._days_to_resolution(end_date_str)
        if days_left is None:
            return None
        if days_left < self.config.min_days_to_resolution:
            return None
        if days_left > self.config.max_days_to_resolution:
            return None
        stats["passed_time"] += 1

        if yes_price >= 0.95 or yes_price <= 0.05:
            return None
        stats["passed_price"] += 1

        if len(q.strip()) < 10:
            return None

        # UPGRADE 4: category specialization filter
        if self.config.focus_categories:
            if not any(fc.lower() in category or fc.lower() in q.lower()
                       for fc in self.config.focus_categories):
                return None
        stats["passed_category"] += 1

        volume_score      = min(volume / 100_000, 1.0)
        liquidity_score   = min(liquidity / 50_000, 1.0)
        uncertainty_score = 1.0 - abs(yes_price - 0.5) * 2
        time_score        = self._time_score(days_left)

        filter_score = (
            volume_score      * 0.35 +
            liquidity_score   * 0.25 +
            uncertainty_score * 0.25 +
            time_score        * 0.15
        )

        return {
            **market,
            "days_left": days_left,
            "filter_score": round(filter_score, 4),
            "filter_metadata": {
                "volume_score":      round(volume_score, 3),
                "liquidity_score":   round(liquidity_score, 3),
                "uncertainty_score": round(uncertainty_score, 3),
                "time_score":        round(time_score, 3),
            },
        }

    def _days_to_resolution(self, end_date_str: str) -> Optional[float]:
        if not end_date_str:
            return None
        try:
            # Handle various ISO formats
            end_date_str = end_date_str.rstrip("Z").split(".")[0]
            end = datetime.fromisoformat(end_date_str)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (end - now).total_seconds() / 86400)
        except Exception:
            return None

    def _time_score(self, days_left: float) -> float:
        """
        Peaks around 7–21 days: enough time for information to arrive,
        not so long that anything can happen.
        """
        sweet_spot = 14.0
        distance = abs(days_left - sweet_spot)
        return max(0.0, 1.0 - distance / sweet_spot)

    def _log_stats(self, stats: dict):
        log.info(
            f"Filter: "
            f"{stats['total']} total → "
            f"{stats['passed_liquidity']} liquidity → "
            f"{stats['passed_volume']} volume → "
            f"{stats['passed_time']} time → "
            f"{stats['passed_price']} price → "
            f"{stats['passed_category']} category → "
            f"{stats['flagged']} flagged"
        )
