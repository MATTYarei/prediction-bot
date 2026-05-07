# PredBot — Prediction Market Trading Bot

5-step agentic pipeline: **Research → Filter → Predict → Execute → Learn**

---

## Architecture

```
predbot/
├── main.py                    # Entry point
├── core/
│   ├── config.py              # All settings (env vars)
│   ├── database.py            # SQLite persistence
│   └── pipeline.py            # Orchestrates all 5 steps
└── agents/
    ├── step1_research.py      # Twitter + Reddit + RSS + Sentiment
    ├── step2_filter.py        # Liquidity / volume / time filter
    ├── step3_predict.py       # XGBoost + Claude LLM calibration
    ├── step4_execute.py       # Kelly sizing + Risk gate + On-chain order
    └── step5_learn.py         # 5-agent post-mortem + model retraining
```

---

## Step 1 · Research (parallel agents)

Four agents run concurrently via `asyncio.gather`:
- **TwitterAgent** — searches recent tweets for top market queries
- **RedditAgent** — scrapes r/politics, r/PredictIt, r/Betting, etc.
- **RSSAgent** — parses BBC, Reuters, NYT, Politico feeds
- **PolymarketFetcher** — pulls up to 300 live markets from the CLOB API
- **SentimentAgent** — Claude scores overall narrative vs. market odds

## Step 2 · Filter

Applies 5 sequential filters across all fetched markets:
1. **Liquidity** ≥ $5,000 (configurable)
2. **Volume** ≥ $500/day
3. **Time to resolution**: 2–60 days
4. **Price not at extremes**: 5%–95% only
5. **Question quality**: non-empty, non-trivial

Remaining markets scored and ranked by a weighted composite of volume, liquidity, uncertainty, and time-window fit.

## Step 3 · Predict (XGBoost + LLM ensemble)

1. **XGBoost** estimates base probability from 6 market features
   - Falls back to calibrated heuristic until 50 resolved trades exist
   - Auto-retrains every 50 new resolutions
2. **Claude LLM** reads question + news context → outputs probability + confidence + reasoning
3. **Ensemble**: weighted average (LLM weight scales with confidence: low=30%, medium=60%, high=80%)
4. **Edge** = calibrated_prob − market_price

## Step 4 · Execute

1. **Kelly Criterion** sizes the bet (capped at 25% Kelly fraction, $100 max)
2. **Risk agent** runs 5 independent checks:
   - Edge ≥ threshold
   - Confidence not "low"
   - Size within bounds
   - Total portfolio exposure < 50% of bankroll
   - No duplicate position on same market
3. **Polymarket CLOB** API places the order (or logs dry-run)
4. Settlement watcher closes trades when markets resolve

## Step 5 · Learn (5 post-mortem agents)

Run concurrently on every resolved trade:
- **CalibrationAgent** — Brier score + log-loss
- **NarrativeAgent** (Claude) — what we got right/wrong, bias detection
- **SignalAgent** — were Twitter/Reddit signals predictive?
- **ModelAgent** — when to retrain XGBoost, prompt adjustments
- **RiskAgent** — Kelly sizing review, threshold adjustments

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run (dry run by default)
python main.py

# 4. To run on a schedule (every 30 min)
watch -n 1800 python main.py
# Or use cron:
# */30 * * * * cd /path/to/predbot && python main.py >> logs/predbot.log 2>&1
```

## Required API Keys

| Key | Where to get | Required? |
|-----|-------------|-----------|
| `ANTHROPIC_API_KEY` | console.anthropic.com | ✅ Yes |
| `POLYMARKET_API_KEY` | polymarket.com/api | ✅ Yes |
| `TWITTER_BEARER_TOKEN` | developer.twitter.com | Optional |
| `REDDIT_CLIENT_ID/SECRET` | reddit.com/prefs/apps | Optional |
| `NEWSAPI_KEY` | newsapi.org | Optional |

The bot runs without Twitter/Reddit/NewsAPI keys — it will use RSS feeds and Polymarket data only.

## Going Live

1. Start with `DRY_RUN=true` and run for at least 2 weeks
2. Review the `learnings` table and calibration stats
3. Set `MAX_POSITION_USD=10` and `DRY_RUN=false` for first live week
4. Scale up position sizes only after Brier score < 0.15 consistently

## Database Tables

- `markets` — all fetched markets
- `predictions` — every probability estimate we made
- `trades` — every order placed (open and closed)
- `learnings` — post-mortem reports from all 5 agents
- `news_cache` — recent news items

## Key Metrics to Watch

- **Brier Score** — below 0.15 is good; above 0.25 means recalibrate
- **Win Rate** — should converge to ~55%+ if edge is real
- **Average Edge** — if declining, markets are becoming more efficient
- **Kelly Growth** — track bankroll curve weekly
