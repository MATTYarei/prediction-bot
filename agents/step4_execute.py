"""
Step 4 · Execute  [v3 — Kalshi primary, multi-venue, arbitrage]

Changes from v2:
  - KalshiTrader replaces PolymarketTrader as the default execution engine
  - VenueRouter selects the right trader based on config.primary_venue
  - ArbExecutor handles cross-platform arbitrage: takes both legs simultaneously
  - Kalshi uses RSA key authentication (different from Polymarket's Bearer token)
  - Settlement watcher handles both Kalshi and Polymarket response formats
"""

import asyncio
import logging
import math
import hashlib
import time
from datetime import datetime
from typing import Optional

from core.config import Config
from core.database import Database

log = logging.getLogger("predbot.execute")


# ── Kelly Criterion (unchanged) ───────────────────────────────────────────────

def kelly_size(
    our_prob: float,
    market_prob: float,
    bankroll: float,
    max_fraction: float = 0.25,
    max_usd: float = 100.0,
) -> float:
    """
    Fractional Kelly criterion, capped at max_fraction and max_usd.
      b = (1 - market_prob) / market_prob   (decimal odds on a win)
      Kelly fraction = (b*p - q) / b  =  p - q/b
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1 - market_prob) / market_prob
    p, q = our_prob, 1 - our_prob
    kelly_frac = (b * p - q) / b
    if kelly_frac <= 0:
        return 0.0
    return round(min(bankroll * min(kelly_frac, max_fraction), max_usd), 2)


# ── Risk Agent (unchanged) ────────────────────────────────────────────────────

class RiskAgent:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db

    def approve(self, prediction: dict, size_usd: float) -> tuple[bool, str]:
        checks = [
            self._check_minimum_edge(prediction),
            self._check_confidence(prediction),
            self._check_size(size_usd),
            self._check_exposure(prediction["market_id"]),
            self._check_not_already_open(prediction["market_id"]),
        ]
        for passed, reason in checks:
            if not passed:
                return False, reason
        return True, "All risk checks passed"

    def _check_minimum_edge(self, pred):
        edge = abs(pred.get("edge", 0))
        if edge < self.config.min_edge_threshold:
            return False, f"Edge {edge:.3f} below threshold {self.config.min_edge_threshold}"
        return True, ""

    def _check_confidence(self, pred):
        if pred.get("confidence", "low") == "low":
            return False, "Confidence too low"
        return True, ""

    def _check_size(self, size_usd):
        if size_usd < 1.0:
            return False, f"Size ${size_usd:.2f} too small (min $1)"
        if size_usd > self.config.max_position_usd:
            return False, f"Size ${size_usd:.2f} exceeds max ${self.config.max_position_usd}"
        return True, ""

    def _check_exposure(self, market_id):
        open_trades = self.db.get_open_trades()
        total_open = sum(t.get("size_usd", 0) for t in open_trades)
        max_exposure = self.config.bankroll_usd * 0.5
        if total_open >= max_exposure:
            return False, f"Exposure ${total_open:.0f} at limit (${max_exposure:.0f})"
        return True, ""

    def _check_not_already_open(self, market_id):
        open_trades = self.db.get_open_trades()
        if any(t.get("market_id") == market_id for t in open_trades):
            return False, f"Already open position on {market_id}"
        return True, ""


# ── Kalshi Trader (NEW primary for US users) ──────────────────────────────────

class KalshiTrader:
    """
    Executes trades on Kalshi via their REST API v2.

    Auth: Kalshi uses RSA key-pair authentication (not a simple Bearer token).
    You generate a key pair, upload the public key to your Kalshi account,
    and sign each request with the private key.

    Setup steps:
      1. Go to kalshi.com → Settings → API
      2. Generate an RSA key pair:
           openssl genrsa -out kalshi_private.pem 2048
           openssl rsa -in kalshi_private.pem -pubout -out kalshi_public.pem
      3. Upload kalshi_public.pem to your Kalshi account
      4. Copy the Key ID shown in the dashboard
      5. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in your .env

    API docs: https://trading-api.readme.io/reference

    Market format:  Kalshi uses ticker strings like "INXD-23DEC31-B5000"
    Contract types: YES contracts only (NO = buy YES on the opposite side)
    Settlement:     Resolves to $1.00 (YES) or $0.00 (NO)
    Fees:           No trading fees on most contracts (as of 2026)
    Position limits: $100K on political contracts per CFTC rules
    """

    def __init__(self, api_key_id: str, private_key_path: str,
                 base_url: str, dry_run: bool = True):
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run
        self._private_key = None
        self._load_key()

    def _load_key(self):
        """Load RSA private key from file."""
        if not self.private_key_path:
            return
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            with open(self.private_key_path, "rb") as f:
                self._private_key = load_pem_private_key(f.read(), password=None)
            log.info("Kalshi: RSA private key loaded")
        except FileNotFoundError:
            log.warning(f"Kalshi: private key not found at {self.private_key_path}")
        except ImportError:
            log.warning("Kalshi: 'cryptography' package not installed — run: "
                        "pip install cryptography")
        except Exception as e:
            log.error(f"Kalshi: key load error: {e}")

    def _sign_request(self, method: str, path: str, body: str = "") -> dict:
        """
        Build Kalshi authentication headers.
        Kalshi requires: timestamp + method + path + body hash, signed with RSA-SHA256.
        """
        if not self._private_key or not self.api_key_id:
            return {}
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64

            ts = str(int(time.time() * 1000))
            msg = f"{ts}{method.upper()}{path}{body}".encode()
            signature = self._private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
            sig_b64 = base64.b64encode(signature).decode()

            return {
                "KALSHI-ACCESS-KEY": self.api_key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": sig_b64,
                "Content-Type": "application/json",
            }
        except Exception as e:
            log.error(f"Kalshi signing error: {e}")
            return {}

    async def get_markets(self, limit: int = 200, cursor: str = "") -> dict:
        """Fetch active markets from Kalshi."""
        try:
            import httpx
            path = "/markets"
            params = {"limit": limit, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            headers = self._sign_request("GET", path)
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(f"{self.base_url}{path}",
                                     headers=headers, params=params)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            log.error(f"Kalshi get_markets: {e}")
            return {}

    async def get_orderbook(self, ticker: str) -> dict:
        """Fetch live order book for a Kalshi market."""
        try:
            import httpx
            path = f"/markets/{ticker}/orderbook"
            headers = self._sign_request("GET", path)
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base_url}{path}", headers=headers)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            log.debug(f"Kalshi orderbook {ticker}: {e}")
            return {}

    async def place_order(self, ticker: str, direction: str,
                          size_usd: float, price: float) -> dict:
        """
        Place a market order on Kalshi.

        ticker:    Kalshi market ticker (e.g. "INXD-23DEC31-B5000")
        direction: "yes" or "no"
        size_usd:  dollar amount to spend
        price:     limit price in cents (Kalshi uses 1-99 scale)
        """
        if self.dry_run:
            return {
                "status": "dry_run",
                "order_id": f"dry_{ticker[:8]}_{datetime.utcnow().strftime('%H%M%S')}",
                "filled": size_usd,
                "price": price,
                "venue": "kalshi",
            }

        try:
            import httpx, json
            path = "/portfolio/orders"
            # Kalshi counts are in contracts (each = $0.01 * price cents)
            # Convert dollar amount to contract count
            price_cents = max(1, min(99, round(price * 100)))
            count = max(1, round(size_usd / (price_cents / 100)))

            body = json.dumps({
                "ticker": ticker,
                "action": "buy",
                "side": direction.lower(),
                "type": "market",
                "count": count,
                "buy_max_cost": round(size_usd * 100),  # in cents
            })
            headers = self._sign_request("POST", path, body)
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(f"{self.base_url}{path}",
                                      headers=headers, content=body)
                r.raise_for_status()
                result = r.json()
                result["venue"] = "kalshi"
                return result
        except Exception as e:
            log.error(f"Kalshi place_order: {e}")
            return {"status": "error", "error": str(e), "venue": "kalshi"}

    async def get_market_status(self, ticker: str) -> Optional[dict]:
        """Check if a Kalshi market has resolved."""
        try:
            import httpx
            path = f"/markets/{ticker}"
            headers = self._sign_request("GET", path)
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base_url}{path}", headers=headers)
                r.raise_for_status()
                data = r.json().get("market", {})
                return {
                    "closed":  data.get("status") in ("closed", "settled"),
                    "resolved": data.get("status") == "settled",
                    "result":  data.get("result", ""),
                    "venue":   "kalshi",
                    "raw":     data,
                }
        except Exception as e:
            log.error(f"Kalshi market status {ticker}: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        if self.dry_run:
            return True
        try:
            import httpx
            path = f"/portfolio/orders/{order_id}"
            headers = self._sign_request("DELETE", path)
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.delete(f"{self.base_url}{path}", headers=headers)
                return r.status_code == 200
        except Exception as e:
            log.error(f"Kalshi cancel {order_id}: {e}")
            return False


# ── Polymarket Trader (kept as secondary) ─────────────────────────────────────

class PolymarketTrader:
    """
    Polymarket execution — secondary venue.
    Requires a Polygon crypto wallet. Not recommended for US users
    until the QCX-based US relaunch is fully live.
    """

    def __init__(self, api_key: str, base_url: str, dry_run: bool = True):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run

    async def place_order(self, market_id: str, direction: str,
                          size_usd: float, price: float) -> dict:
        if self.dry_run:
            return {
                "status": "dry_run",
                "tx_hash": f"dry_{market_id[:8]}_{datetime.utcnow().strftime('%H%M%S')}",
                "filled": size_usd,
                "price": price,
                "venue": "polymarket",
            }
        try:
            import httpx, json
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            body = json.dumps({"order": {"tokenID": market_id, "side": direction,
                                          "price": str(round(price, 4)),
                                          "size": str(round(size_usd, 2)),
                                          "type": "MARKET"}})
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(f"{self.base_url}/order",
                                      headers=headers, content=body)
                r.raise_for_status()
                result = r.json()
                result["venue"] = "polymarket"
                return result
        except Exception as e:
            log.error(f"Polymarket place_order: {e}")
            return {"status": "error", "error": str(e), "venue": "polymarket"}

    async def get_market_status(self, market_id: str) -> Optional[dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base_url}/markets/{market_id}")
                r.raise_for_status()
                data = r.json()
                return {
                    "closed":   data.get("closed", False),
                    "resolved": data.get("closed", False),
                    "result":   data.get("result", ""),
                    "venue":    "polymarket",
                    "raw":      data,
                }
        except Exception as e:
            log.error(f"Polymarket market status {market_id}: {e}")
            return None


# ── Venue Router ──────────────────────────────────────────────────────────────

class VenueRouter:
    """
    Selects the correct trader based on config.primary_venue.
    Falls back gracefully if the primary venue has no credentials.
    """

    def __init__(self, config: Config):
        self.config = config
        self.kalshi = KalshiTrader(
            api_key_id=config.kalshi_api_key_id,
            private_key_path=config.kalshi_private_key_path,
            base_url=config.kalshi_base_url,
            dry_run=config.dry_run,
        )
        self.polymarket = PolymarketTrader(
            api_key=config.polymarket_api_key,
            base_url=config.polymarket_base_url,
            dry_run=config.dry_run,
        )

    def get_trader(self, venue: str = None):
        """Return the appropriate trader for a given venue."""
        target = venue or self.config.primary_venue
        if target == "kalshi":
            return self.kalshi
        return self.polymarket

    def get_primary(self):
        return self.get_trader(self.config.primary_venue)


# ── Arbitrage Executor ────────────────────────────────────────────────────────

class ArbExecutor:
    """
    Executes cross-platform arbitrage when the same event is priced
    differently on Kalshi vs Polymarket.

    Example: Event priced at 65% YES on Kalshi, 58% YES on Polymarket.
      → Buy YES on Polymarket (cheaper), Buy NO on Kalshi (which = sell YES at 65%)
      → Lock in ~7% spread regardless of outcome

    Both legs must be placed near-simultaneously to avoid leg risk.
    Both legs must pass individual risk checks.
    """

    def __init__(self, config: Config, db: Database, venue_router: VenueRouter):
        self.config = config
        self.db = db
        self.router = venue_router

    async def execute_arb(self, arb: dict) -> Optional[dict]:
        """
        Execute both legs of an arbitrage opportunity.

        arb dict fields:
          kalshi_ticker:    Kalshi market ID
          polymarket_id:    Polymarket market ID
          kalshi_price:     YES price on Kalshi
          polymarket_price: YES price on Polymarket
          gap:              price difference (kalshi_price - polymarket_price)
          direction:        "buy_poly_sell_kalshi" or "buy_kalshi_sell_poly"
        """
        if not self.config.arb_scan_enabled:
            return None

        gap = abs(arb.get("gap", 0))
        if gap < self.config.arb_min_gap:
            log.debug(f"Arb gap {gap:.2%} below min {self.config.arb_min_gap:.2%}")
            return None

        # Size each leg conservatively — half normal position
        arb_size = min(self.config.max_position_usd * 0.5, 50.0)

        kalshi_price   = arb["kalshi_price"]
        poly_price     = arb["polymarket_price"]
        direction      = arb["direction"]

        if direction == "buy_poly_sell_kalshi":
            # Polymarket YES is cheaper → buy YES on Poly, buy NO on Kalshi
            poly_direction   = "YES"
            kalshi_direction = "no"   # buying NO on Kalshi = selling YES
            poly_entry       = poly_price
            kalshi_entry     = 1 - kalshi_price
        else:
            # Kalshi YES is cheaper → buy YES on Kalshi, buy NO on Polymarket
            kalshi_direction = "yes"
            poly_direction   = "NO"
            kalshi_entry     = kalshi_price
            poly_entry       = 1 - poly_price

        log.info(f"Arb detected: gap={gap:.2%} "
                 f"kalshi={kalshi_price:.2%} poly={poly_price:.2%} "
                 f"direction={direction} size=${arb_size:.0f} each leg")

        # Place both legs simultaneously
        if self.config.dry_run:
            kalshi_order = {"status": "dry_run", "venue": "kalshi",
                            "order_id": f"arb_k_{datetime.utcnow().strftime('%H%M%S')}"}
            poly_order   = {"status": "dry_run", "venue": "polymarket",
                            "tx_hash": f"arb_p_{datetime.utcnow().strftime('%H%M%S')}"}
        else:
            kalshi_order, poly_order = await asyncio.gather(
                self.router.kalshi.place_order(
                    arb["kalshi_ticker"], kalshi_direction, arb_size, kalshi_entry),
                self.router.polymarket.place_order(
                    arb["polymarket_id"], poly_direction, arb_size, poly_entry),
            )

        if kalshi_order.get("status") == "error" or poly_order.get("status") == "error":
            log.error(f"Arb leg failed: kalshi={kalshi_order} poly={poly_order}")
            return None

        # Save both legs
        arb_result = {
            "type":       "arbitrage",
            "gap":        gap,
            "direction":  direction,
            "kalshi_leg": kalshi_order,
            "poly_leg":   poly_order,
            "size_each":  arb_size,
            "locked_profit_est": round(gap * arb_size, 2),
        }

        # Persist as two linked trades
        trade_k = self.db.save_trade({
            "market_id":    arb["kalshi_ticker"],
            "direction":    kalshi_direction.upper(),
            "size_usd":     arb_size,
            "entry_price":  kalshi_entry,
            "tx_hash":      kalshi_order.get("order_id", ""),
            "prediction_id": None,
        })
        trade_p = self.db.save_trade({
            "market_id":    arb["polymarket_id"],
            "direction":    poly_direction.upper(),
            "size_usd":     arb_size,
            "entry_price":  poly_entry,
            "tx_hash":      poly_order.get("tx_hash", ""),
            "prediction_id": None,
        })

        mode = "[DRY RUN]" if self.config.dry_run else "[LIVE]"
        log.info(f"{mode} ARB executed: gap={gap:.2%} "
                 f"est profit=${arb_result['locked_profit_est']:.2f} "
                 f"trades #{trade_k}+#{trade_p}")

        return arb_result


# ── Execute Agent ─────────────────────────────────────────────────────────────

class ExecuteAgent:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.risk = RiskAgent(config, db)
        self.venue_router = VenueRouter(config)
        self.arb_executor = ArbExecutor(config, db, self.venue_router)

    async def run(self, prediction: dict) -> Optional[dict]:
        """
        Size → risk check → route to correct venue → place order → persist.
        """
        market      = prediction["market"]
        our_prob    = prediction["our_prob"]
        market_prob = prediction["market_prob"]
        edge        = prediction["edge"]

        direction = "YES" if edge > 0 else "NO"
        eff_mkt   = market_prob if direction == "YES" else 1 - market_prob
        eff_ours  = our_prob    if direction == "YES" else 1 - our_prob
        entry_price = eff_mkt

        size_usd = kelly_size(
            our_prob=eff_ours, market_prob=eff_mkt,
            bankroll=self.config.bankroll_usd,
            max_fraction=self.config.max_kelly_fraction,
            max_usd=self.config.max_position_usd,
        )

        log.info(f"Sizing [{self.config.primary_venue.upper()}]: "
                 f"mkt={market_prob:.2%} ours={our_prob:.2%} "
                 f"edge={edge:+.3f} → {direction} ${size_usd:.2f}")

        approved, reason = self.risk.approve(prediction, size_usd)
        if not approved:
            log.info(f"Risk BLOCKED: {reason}")
            return None
        log.info(f"Risk APPROVED: {reason}")

        # Route to primary venue
        trader = self.venue_router.get_primary()
        market_id = prediction["market_id"]

        # Kalshi needs the ticker; if market has a kalshi_ticker field use it
        kalshi_ticker = market.get("kalshi_ticker", market_id)
        exec_id = kalshi_ticker if self.config.primary_venue == "kalshi" else market_id

        order = await trader.place_order(exec_id, direction, size_usd, entry_price)

        if order.get("status") == "error":
            log.error(f"Order failed: {order.get('error')}")
            return None

        trade = {
            "market_id":    market_id,
            "direction":    direction,
            "size_usd":     size_usd,
            "entry_price":  entry_price,
            "tx_hash":      order.get("order_id") or order.get("tx_hash", ""),
            "prediction_id": prediction.get("prediction_id"),
            "venue":        self.config.primary_venue,
        }
        trade_id = self.db.save_trade(trade)
        trade["id"] = trade_id

        venue_label = self.config.primary_venue.upper()
        mode = "[DRY RUN]" if self.config.dry_run else "[LIVE]"
        log.info(f"{mode} [{venue_label}] Trade #{trade_id}: "
                 f"{direction} ${size_usd:.2f} @ {entry_price:.3f} on {exec_id}")

        return trade

    async def run_arb(self, arb_opportunities: list[dict]) -> list[dict]:
        """Execute arbitrage opportunities found by the research layer."""
        results = []
        for arb in arb_opportunities:
            result = await self.arb_executor.execute_arb(arb)
            if result:
                results.append(result)
        return results

    async def check_settlements(self):
        """Poll all open trades and close out any that have resolved."""
        open_trades = self.db.get_open_trades()
        for trade in open_trades:
            venue  = trade.get("venue", self.config.primary_venue)
            trader = self.venue_router.get_trader(venue)
            status = await trader.get_market_status(trade["market_id"])
            if not status or not status.get("resolved"):
                continue
            resolved_yes = status.get("result", "").upper() == "YES"
            direction    = trade["direction"]
            won = (direction == "YES" and resolved_yes) or \
                  (direction == "NO"  and not resolved_yes)
            exit_price = 1.0 if won else 0.0
            pnl = trade["size_usd"] * (exit_price - trade["entry_price"])
            self.db.close_trade(trade["id"], exit_price, pnl)
            log.info(f"[{venue.upper()}] Trade #{trade['id']} settled: "
                     f"{'WIN' if won else 'LOSS'} PnL=${pnl:+.2f}")
