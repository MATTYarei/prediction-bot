"""
Config — loads all settings from environment variables.
Copy .env.example to .env and fill in your keys.

v3 changes:
  - Kalshi is now the PRIMARY execution venue (US-regulated, CFTC-licensed)
  - Prediction Hunt added as unified cross-platform research API
  - Polymarket kept as secondary/research-only (crypto wallet required for execution)
  - primary_venue setting controls which platform executes trades
  - arb_scan_enabled: scan for price gaps between Kalshi and Polymarket
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Anthropic ──────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    # ── Primary execution venue ────────────────────────────────
    # "kalshi"     → US-regulated, CFTC-licensed, fiat deposits (recommended for US)
    # "polymarket" → Global liquidity, crypto-only, offshore (non-US or with wallet)
    primary_venue: str = "kalshi"

    # ── Kalshi (primary for US users) ──────────────────────────
    kalshi_api_key: str = ""
    kalshi_api_key_id: str = ""          # RSA key ID for Kalshi auth
    kalshi_private_key_path: str = ""    # path to RSA private key .pem file
    kalshi_base_url: str = "https://trading-api.kalshi.com/trade-api/v2"

    # ── Polymarket (secondary / global research) ───────────────
    polymarket_api_key: str = ""
    polymarket_base_url: str = "https://clob.polymarket.com"

    # ── Prediction Hunt (unified cross-platform API) ───────────
    # Covers Kalshi + Polymarket + PredictIt + ProphetX + Opinion in one API
    # Get key at: predictionhunt.com
    prediction_hunt_api_key: str = ""
    prediction_hunt_base_url: str = "https://www.predictionhunt.com/api/v2"

    # ── Arbitrage scanning ─────────────────────────────────────
    # When True, bot scans for same event priced differently on Kalshi vs Polymarket
    arb_scan_enabled: bool = True
    arb_min_gap: float = 0.04            # min price gap to flag as arbitrage (4%)

    # ── Data sources ───────────────────────────────────────────
    twitter_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "PredBot/1.0"
    newsapi_key: str = ""

    # ── Strategy knobs ─────────────────────────────────────────
    min_liquidity_usd: float = 5000.0
    min_daily_volume_usd: float = 500.0
    min_days_to_resolution: int = 2
    max_days_to_resolution: int = 60
    min_edge_threshold: float = 0.05
    max_kelly_fraction: float = 0.25
    max_position_usd: float = 100.0
    bankroll_usd: float = 1000.0

    # ── Tighter execution filters ──────────────────────────────
    high_confidence_only: bool = True
    require_cross_platform_gap: bool = True
    min_signal_strength: float = 0.20

    # ── Niche specialization ───────────────────────────────────
    focus_categories: list = None        # None = trade all

    # ── Database ───────────────────────────────────────────────
    db_path: str = "predbot.db"

    # ── Misc ───────────────────────────────────────────────────
    dry_run: bool = True
    max_markets_to_scan: int = 300
    research_subreddits: list = field(default_factory=lambda: [
        "politics", "PredictIt", "Betting", "economy",
        "worldnews", "sports", "technology"
    ])
    rss_feeds: list = field(default_factory=lambda: [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://feeds.reuters.com/reuters/topNews",
        "https://www.politico.com/rss/politicopicks.xml",
    ])

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            primary_venue=os.getenv("PRIMARY_VENUE", "kalshi"),
            kalshi_api_key=os.getenv("KALSHI_API_KEY", ""),
            kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
            kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
            kalshi_base_url=os.getenv("KALSHI_BASE_URL",
                                       "https://trading-api.kalshi.com/trade-api/v2"),
            polymarket_api_key=os.getenv("POLYMARKET_API_KEY", ""),
            polymarket_base_url=os.getenv("POLYMARKET_BASE_URL",
                                           "https://clob.polymarket.com"),
            prediction_hunt_api_key=os.getenv("PREDICTION_HUNT_API_KEY", ""),
            arb_scan_enabled=os.getenv("ARB_SCAN_ENABLED", "true").lower() == "true",
            arb_min_gap=float(os.getenv("ARB_MIN_GAP", "0.04")),
            twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            newsapi_key=os.getenv("NEWSAPI_KEY", ""),
            min_liquidity_usd=float(os.getenv("MIN_LIQUIDITY_USD", "5000")),
            min_daily_volume_usd=float(os.getenv("MIN_DAILY_VOLUME_USD", "500")),
            min_edge_threshold=float(os.getenv("MIN_EDGE_THRESHOLD", "0.05")),
            max_kelly_fraction=float(os.getenv("MAX_KELLY_FRACTION", "0.25")),
            max_position_usd=float(os.getenv("MAX_POSITION_USD", "100")),
            bankroll_usd=float(os.getenv("BANKROLL_USD", "1000")),
            dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
            db_path=os.getenv("DB_PATH", "predbot.db"),
            high_confidence_only=os.getenv("HIGH_CONFIDENCE_ONLY", "true").lower() == "true",
            require_cross_platform_gap=os.getenv("REQUIRE_CROSS_PLATFORM_GAP",
                                                   "true").lower() == "true",
            min_signal_strength=float(os.getenv("MIN_SIGNAL_STRENGTH", "0.20")),
            focus_categories=os.getenv("FOCUS_CATEGORIES", "").split(",")
                             if os.getenv("FOCUS_CATEGORIES") else None,
        )
