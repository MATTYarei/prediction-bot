"""
Base Rate Library  [NEW — Upgrade 2]

A structured knowledge base of historical base rates for common prediction
market categories. Claude is anchored to these numbers BEFORE reading news,
preventing recency bias and availability heuristic errors.

Why this works: humans (and LLMs) are notoriously bad at base rates.
When asked "will the Fed cut rates?" after 3 bullish headlines, the LLM
wants to say 80%. But historically, the Fed cuts at a rate consistent
with ~35% in any given meeting cycle. Anchoring first, then adjusting
for current evidence, produces much better calibrated probabilities.

Format: each entry has a base_rate (float 0-1), context, and adjustment_factors.
"""

BASE_RATES = {

    # ── Economics / Federal Reserve ───────────────────────────────────────────
    "fed_rate_cut_single_meeting": {
        "base_rate": 0.28,
        "description": "Probability Fed cuts rates at any given scheduled FOMC meeting",
        "context": "Since 2000, Fed cuts at ~28% of meetings. Higher when inflation < 2.5% and unemployment rising.",
        "adjustment_factors": [
            "inflation_above_3pct: -0.15",
            "unemployment_rising: +0.12",
            "recession_declared: +0.30",
            "fed_chair_dovish_speech: +0.08",
        ],
        "categories": ["economics", "fed", "interest_rates"],
    },
    "fed_rate_hike_single_meeting": {
        "base_rate": 0.22,
        "description": "Probability Fed hikes rates at any given FOMC meeting",
        "context": "Since 2000, Fed hikes at ~22% of meetings. Strongly correlated with CPI > 3%.",
        "categories": ["economics", "fed", "interest_rates"],
    },
    "us_recession_next_12_months": {
        "base_rate": 0.17,
        "description": "Base rate of US recession starting in any given 12-month window",
        "context": "US has ~6 recessions per 35 years = ~17% per year. Higher when yield curve inverted >6 months.",
        "adjustment_factors": [
            "yield_curve_inverted_6mo: +0.12",
            "unemployment_rising_3mo: +0.08",
            "gdp_negative_1q: +0.20",
        ],
        "categories": ["economics", "recession"],
    },

    # ── Crypto ────────────────────────────────────────────────────────────────
    "btc_new_ath_within_90_days": {
        "base_rate": 0.18,
        "description": "BTC reaches new all-time high within any 90-day window",
        "context": "BTC hits new ATH in roughly 18% of 90-day periods historically. Much higher in bull years.",
        "adjustment_factors": [
            "post_halving_year: +0.20",
            "btc_within_10pct_ath: +0.25",
            "crypto_fear_greed_extreme_greed: +0.10",
        ],
        "categories": ["crypto", "bitcoin"],
    },
    "btc_drop_20pct_from_high": {
        "base_rate": 0.35,
        "description": "BTC drops >20% from local high within 60 days",
        "context": "BTC experiences >20% drawdowns roughly 35% of the time in any 60-day window.",
        "categories": ["crypto", "bitcoin"],
    },

    # ── Politics / Elections ──────────────────────────────────────────────────
    "incumbent_wins_reelection": {
        "base_rate": 0.63,
        "description": "Incumbent president wins reelection when running",
        "context": "US incumbents win ~63% of elections when seeking reelection (1900-2024).",
        "adjustment_factors": [
            "approval_below_45pct: -0.15",
            "recession_in_election_year: -0.18",
            "gdp_growth_above_3pct: +0.10",
        ],
        "categories": ["politics", "elections"],
    },
    "senate_bill_passes_from_committee": {
        "base_rate": 0.14,
        "description": "Bill that clears Senate committee becomes law",
        "context": "Only ~14% of bills that clear committee ultimately pass both chambers and are signed.",
        "categories": ["politics", "legislation"],
    },

    # ── Geopolitics ───────────────────────────────────────────────────────────
    "peace_deal_signed_active_conflict": {
        "base_rate": 0.08,
        "description": "Active military conflict ends with formal peace deal within 12 months",
        "context": "Active conflicts in the modern era resolve via peace deal in only ~8% of 12-month windows.",
        "categories": ["geopolitics", "conflict"],
    },
    "un_sanctions_lifted_within_year": {
        "base_rate": 0.11,
        "description": "Existing UN sanctions on a country are lifted within 12 months",
        "context": "UN sanctions, once imposed, are lifted within a year only ~11% of the time.",
        "categories": ["geopolitics", "sanctions"],
    },

    # ── Finance / Markets ─────────────────────────────────────────────────────
    "sp500_positive_year": {
        "base_rate": 0.72,
        "description": "S&P 500 ends calendar year higher than it started",
        "context": "S&P 500 has positive returns in ~72% of calendar years since 1928.",
        "categories": ["finance", "stocks", "sp500"],
    },
    "sp500_correction_20pct": {
        "base_rate": 0.23,
        "description": "S&P 500 corrects 20%+ from high in any given 12-month window",
        "context": "Bear markets (>20% decline) occur roughly every 4-5 years = ~23% per year.",
        "categories": ["finance", "stocks"],
    },

    # ── Technology ────────────────────────────────────────────────────────────
    "tech_ipo_positive_first_day": {
        "base_rate": 0.71,
        "description": "Tech IPO closes higher than offering price on day 1",
        "context": "~71% of tech IPOs close above offering price on day 1 (2010-2024 data).",
        "categories": ["technology", "ipo"],
    },
    "ai_model_beats_benchmark": {
        "base_rate": 0.55,
        "description": "New major AI model release beats previous SOTA on headline benchmark",
        "context": "Since 2020, new major releases from top labs beat prior SOTA ~55% of the time on their chosen benchmark.",
        "categories": ["technology", "ai"],
    },
}


def get_relevant_base_rates(question: str, category: str = "") -> list[dict]:
    """
    Return base rates relevant to a market question.
    Matches on category tag and keyword search in question text.
    """
    question_lower = question.lower()
    results = []

    for key, data in BASE_RATES.items():
        relevance = 0

        # Category match
        if category:
            if any(c in category.lower() for c in data.get("categories", [])):
                relevance += 2

        # Keyword match against description + context
        searchable = (data["description"] + " " + data.get("context", "")).lower()
        question_words = set(question_lower.split()) - {"will", "the", "a", "an", "be",
                                                         "in", "of", "to", "is", "by"}
        keyword_hits = sum(1 for w in question_words if len(w) > 4 and w in searchable)
        relevance += keyword_hits

        if relevance >= 2:
            results.append({
                "key": key,
                "base_rate": data["base_rate"],
                "description": data["description"],
                "context": data["context"],
                "adjustment_factors": data.get("adjustment_factors", []),
                "relevance_score": relevance,
            })

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:3]  # top 3 most relevant


def format_for_prompt(base_rates: list[dict]) -> str:
    """Format base rates into a clear prompt block for Claude."""
    if not base_rates:
        return "No specific base rates available for this question type."
    lines = ["HISTORICAL BASE RATES (anchor to these before adjusting for current news):"]
    for br in base_rates:
        lines.append(f"\n  {br['description']}")
        lines.append(f"  Base rate: {br['base_rate']:.0%}")
        lines.append(f"  Context: {br['context']}")
        if br.get("adjustment_factors"):
            lines.append("  Key adjustments:")
            for adj in br["adjustment_factors"][:3]:
                lines.append(f"    • {adj}")
    return "\n".join(lines)
