"""
PredBot — Prediction Market Trading Bot
5-step pipeline: Research → Filter → Predict → Execute → Learn
"""

import asyncio
import logging
from datetime import datetime
from core.pipeline import Pipeline
from core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predbot")


async def main():
    log.info("=" * 60)
    log.info("PredBot starting up")
    log.info(f"Run time: {datetime.utcnow().isoformat()}Z")
    log.info("=" * 60)

    config = Config.from_env()
    pipeline = Pipeline(config)

    await pipeline.run_cycle()


if __name__ == "__main__":
    asyncio.run(main())
