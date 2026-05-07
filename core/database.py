"""
Database — SQLite-backed persistence for trades, predictions, and learnings.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("predbot.db")


class Database:
    def __init__(self, db_path: str = "predbot.db"):
        self.path = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS markets (
                    id TEXT PRIMARY KEY,
                    platform TEXT,
                    question TEXT,
                    yes_price REAL,
                    volume_usd REAL,
                    liquidity_usd REAL,
                    end_date TEXT,
                    raw_json TEXT,
                    fetched_at TEXT
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    our_prob REAL,
                    market_prob REAL,
                    edge REAL,
                    confidence TEXT,
                    reasoning TEXT,
                    xgboost_prob REAL,
                    llm_prob REAL,
                    calibrated_prob REAL,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    direction TEXT,
                    size_usd REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    status TEXT DEFAULT 'open',
                    tx_hash TEXT,
                    prediction_id INTEGER,
                    opened_at TEXT,
                    closed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    market_id TEXT,
                    outcome TEXT,
                    our_prob REAL,
                    actual_outcome INTEGER,
                    brier_score REAL,
                    log_loss REAL,
                    lessons TEXT,
                    agent_reports TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS news_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    headline TEXT,
                    summary TEXT,
                    sentiment REAL,
                    url TEXT,
                    fetched_at TEXT
                );
            """)
        log.info(f"Database ready at {self.path}")

    # ── Markets ────────────────────────────────────────────────

    def upsert_market(self, market: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO markets
                (id, platform, question, yes_price, volume_usd, liquidity_usd, end_date, raw_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market["id"], market["platform"], market["question"],
                market["yes_price"], market.get("volume_usd", 0),
                market.get("liquidity_usd", 0), market.get("end_date", ""),
                json.dumps(market), datetime.utcnow().isoformat()
            ))

    def get_market(self, market_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM markets WHERE id=?", (market_id,)).fetchone()
            return dict(row) if row else None

    # ── Predictions ────────────────────────────────────────────

    def save_prediction(self, pred: dict) -> int:
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO predictions
                (market_id, our_prob, market_prob, edge, confidence, reasoning,
                 xgboost_prob, llm_prob, calibrated_prob, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred["market_id"], pred["our_prob"], pred["market_prob"],
                pred["edge"], pred.get("confidence", "medium"),
                pred.get("reasoning", ""), pred.get("xgboost_prob"),
                pred.get("llm_prob"), pred.get("calibrated_prob"),
                datetime.utcnow().isoformat()
            ))
            return cur.lastrowid

    # ── Trades ─────────────────────────────────────────────────

    def save_trade(self, trade: dict) -> int:
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO trades
                (market_id, direction, size_usd, entry_price, status,
                 tx_hash, prediction_id, opened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["market_id"], trade["direction"], trade["size_usd"],
                trade["entry_price"], "open", trade.get("tx_hash", ""),
                trade.get("prediction_id"), datetime.utcnow().isoformat()
            ))
            return cur.lastrowid

    def close_trade(self, trade_id: int, exit_price: float, pnl: float):
        with self._conn() as conn:
            conn.execute("""
                UPDATE trades SET exit_price=?, pnl=?, status='closed', closed_at=?
                WHERE id=?
            """, (exit_price, pnl, datetime.utcnow().isoformat(), trade_id))

    def get_open_trades(self) -> list:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM trades WHERE status='open'").fetchall()
            return [dict(r) for r in rows]

    # ── Learnings ──────────────────────────────────────────────

    def save_learning(self, learning: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO learnings
                (trade_id, market_id, outcome, our_prob, actual_outcome,
                 brier_score, log_loss, lessons, agent_reports, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                learning.get("trade_id"), learning["market_id"],
                learning.get("outcome", ""), learning.get("our_prob"),
                learning.get("actual_outcome"), learning.get("brier_score"),
                learning.get("log_loss"), learning.get("lessons", ""),
                json.dumps(learning.get("agent_reports", {})),
                datetime.utcnow().isoformat()
            ))

    def get_recent_learnings(self, limit: int = 20) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM learnings ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_calibration_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       AVG(brier_score) as avg_brier,
                       AVG(CASE WHEN actual_outcome=1 THEN 1.0 ELSE 0.0 END) as win_rate
                FROM learnings WHERE brier_score IS NOT NULL
            """).fetchone()
            return dict(row) if row else {}

    # ── News cache ─────────────────────────────────────────────

    def save_news(self, articles: list):
        with self._conn() as conn:
            for a in articles:
                conn.execute("""
                    INSERT OR IGNORE INTO news_cache
                    (source, headline, summary, sentiment, url, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    a.get("source", ""), a.get("headline", ""),
                    a.get("summary", ""), a.get("sentiment", 0.0),
                    a.get("url", ""), datetime.utcnow().isoformat()
                ))

    def get_recent_news(self, hours: int = 6) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM news_cache
                WHERE fetched_at > datetime('now', ? || ' hours')
                ORDER BY fetched_at DESC LIMIT 100
            """, (f"-{hours}",)).fetchall()
            return [dict(r) for r in rows]
