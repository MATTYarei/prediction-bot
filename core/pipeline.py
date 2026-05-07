"""
Pipeline — orchestrates the full 5-step cycle.
Research → Filter → Predict → Execute → Learn
"""

import asyncio
import logging
from datetime import datetime

from core.config import Config
from core.database import Database
from agents.step1_research import ResearchOrchestrator
from agents.step2_filter import FilterAgent
from agents.step3_predict import PredictAgent
from agents.step4_execute import ExecuteAgent
from agents.step5_learn import LearnOrchestrator

log = logging.getLogger("predbot.pipeline")


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.db_path)

        self.research = ResearchOrchestrator(config, self.db)
        self.filter = FilterAgent(config, self.db)
        self.predict = PredictAgent(config, self.db)
        self.execute = ExecuteAgent(config, self.db)
        self.learn = LearnOrchestrator(config, self.db)

    async def run_cycle(self):
        start = datetime.utcnow()
        log.info("━━━━━━━━━━━━ CYCLE START ━━━━━━━━━━━━")

        # ── Step 1: Research ───────────────────────────────────
        log.info("STEP 1 · RESEARCH — parallel scraping")
        research_data = await self.research.run()
        log.info(f"  → {len(research_data['markets'])} markets fetched, "
                 f"{len(research_data['news'])} news items, "
                 f"sentiment score: {research_data['sentiment_score']:.2f}")

        # ── Step 2: Filter ─────────────────────────────────────
        log.info("STEP 2 · FILTER — scanning all markets")
        flagged_markets = await self.filter.run(research_data["markets"])
        log.info(f"  → {len(flagged_markets)} markets flagged from "
                 f"{len(research_data['markets'])} scanned")

        if not flagged_markets:
            log.info("  No tradeable markets found this cycle. Exiting.")
            return

        # ── Step 3: Predict ────────────────────────────────────
        log.info("STEP 3 · PREDICT — base rates + domain prompts + ensemble")
        predictions = []
        signal_packages = research_data.get("signal_packages", {})
        for market in flagged_markets:
            pkg  = signal_packages.get(market["id"], {})
            # Skip low-signal markets early (Upgrade 3)
            if pkg and pkg.get("signal_strength", 1.0) < self.config.min_signal_strength:
                log.info(f"  Skipping {market['id']}: signal_strength too low")
                continue
            pred = await self.predict.run(market, research_data["news"], pkg)
            predictions.append(pred)
            status = "TRADE" if pred.get("tradeable", True) else "SKIP"
            log.info(f"  [{pred.get('category','?')[:3].upper()}] "
                     f"edge={pred['edge']:+.3f} conf={pred['confidence']} {status}")

        # ── Step 4: Execute ────────────────────────────────────
        log.info("STEP 4 · EXECUTE — risk check + trade placement")
        trades = []
        for pred in predictions:
            # Respect execution filter decision from Step 3
            if not pred.get("tradeable", True):
                log.info(f"  SKIP (prediction filter): {pred.get('filter_reason', '')}")
                continue
            if abs(pred["edge"]) >= self.config.min_edge_threshold:
                trade = await self.execute.run(pred)
                if trade:
                    trades.append(trade)
                    log.info(f"  ✓ {'DRY RUN' if self.config.dry_run else 'LIVE'} "
                             f"trade: {trade['direction']} ${trade['size_usd']:.2f} "
                             f"on {trade['market_id']}")
        log.info(f"  → {len(trades)} trades placed")

        # ── Step 4b: Arbitrage ─────────────────────────────────
        arb_opps = research_data.get("arb_opportunities", [])
        if arb_opps:
            log.info(f"STEP 4b · ARB — {len(arb_opps)} opportunities")
            arb_results = await self.execute.run_arb(arb_opps)
            log.info(f"  → {len(arb_results)} arb trades executed")
            trades.extend(arb_results)

        # ── Step 5: Learn (on resolved markets) ───────────────
        log.info("STEP 5 · LEARN — post-mortem on settled trades")
        resolved = await self.learn.run()
        log.info(f"  → {len(resolved)} post-mortems completed")

        # ── Summary ────────────────────────────────────────────
        elapsed = (datetime.utcnow() - start).total_seconds()
        stats = self.db.get_calibration_stats()
        log.info("━━━━━━━━━━━━ CYCLE COMPLETE ━━━━━━━━━━━━")
        log.info(f"  Duration     : {elapsed:.1f}s")
        log.info(f"  Markets seen : {len(research_data['markets'])}")
        log.info(f"  Flagged      : {len(flagged_markets)}")
        log.info(f"  Trades placed: {len(trades)}")
        log.info(f"  Avg Brier    : {stats.get('avg_brier', 'n/a')}")
        log.info(f"  Win rate     : {stats.get('win_rate', 'n/a')}")
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
