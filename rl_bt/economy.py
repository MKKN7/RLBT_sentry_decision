"""Conservative UC purchase accounting reference, not a live referee client."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
from threading import RLock

from rl_bt.uc_rules import rule_facts


def remote_heal_price(remaining_s):
    """PDF p77/table5-7: ceil applies to elapsed/60 * 20, all inside brackets."""
    remaining = Decimal(str(remaining_s))
    if not remaining.is_finite() or not 0 <= remaining <= 420:
        raise ValueError("UC match time must be within 0..420 seconds")
    return 50 + int(((Decimal(420)-remaining) / 60 * 20).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class PurchaseRequest:
    request_id: str
    kind: str
    coins: int
    wallet_version: int
    issued_at_s: float
    ttl_s: float = 2.
    quantity: int = 1


def uc_purchase_quote(*, request_id, kind, quantity, remaining_s, wallet_version,
                      now_s, team_purchased_17mm=None):
    """Validate UC bundle sizes and the known team purchase count before reserving."""
    facts = rule_facts()
    if type(quantity) is not int or quantity <= 0:
        raise ValueError("Positive integer quantity required")
    if kind == "remote_heal":
        if quantity != 1:
            raise ValueError("One heal transaction per request")
        coins = remote_heal_price(remaining_s)
    elif kind in ("remote_ammo", "local_ammo"):
        if type(team_purchased_17mm) is not int:
            raise ValueError("Known team paid-ammo count required; unknown is not zero")
        fact = facts["remote_17mm_exchange" if kind == "remote_ammo" else "local_17mm_exchange"]["value"]
        if quantity % fact["projectiles"]:
            raise ValueError("Quantity must match the UC exchange bundle")
        if team_purchased_17mm < 0 or team_purchased_17mm + quantity > facts["team_purchased_17mm_limit"]["value"]:
            raise ValueError("Team paid-ammo limit exceeded; free/gift ammo is separate")
        coins = quantity // fact["projectiles"] * fact["coins"]
    else:
        raise ValueError("Unknown UC purchase kind")
    return PurchaseRequest(request_id, kind, coins, wallet_version, now_s, quantity=quantity)


class SentryBudgetLedger:
    """Explicit test budgets; configure from team policy before deployment.

    One request in flight, atomic reservations, idempotent acceptance. Timeout
    keeps the reservation until authoritative reconciliation: it is not a denial.
    This offline ledger does not implement live shared-wallet reconciliation.
    """
    TERMINAL = {"effect_confirmed", "failed_no_refund", "denied"}

    def __init__(self, *, team_balance, sentry_match_cap, team_reserve_floor):
        if any(type(v) is not int or v < 0 for v in (team_balance, sentry_match_cap, team_reserve_floor)):
            raise ValueError("Explicit nonnegative integer coin budgets required")
        self.balance = team_balance
        self.cap = sentry_match_cap
        self.floor = team_reserve_floor
        self.spent = 0
        self.version = 0
        self.requests = {}
        self.lock = RLock()

    def reserve(self, request, *, now_s, alive, out_of_combat):
        with self.lock:
            if request.request_id in self.requests:
                return "duplicate_request"
            if request.kind not in ("remote_heal", "remote_ammo", "local_ammo"):
                return "unsupported_purchase"
            if type(request.coins) is not int or request.coins <= 0:
                return "invalid_quote_or_free_resource_not_purchase"
            if not all(math.isfinite(x) for x in (now_s, request.issued_at_s, request.ttl_s)):
                return "invalid_time"
            if request.ttl_s <= 0 or not 0 <= now_s-request.issued_at_s <= request.ttl_s:
                return "stale_quote"
            if request.wallet_version != self.version:
                return "stale_wallet"
            if not alive or (request.kind.startswith("remote") and not out_of_combat):
                return "rule_ineligible"
            if any(x["status"] not in self.TERMINAL for x in self.requests.values()):
                return "purchase_in_flight"
            if self.spent + request.coins > self.cap:
                return "sentry_budget_exceeded"
            if self.balance - request.coins < self.floor:
                return "team_reserve_protected"
            self.requests[request.request_id] = {"request": request, "status": "reserved"}
            return None

    def mark_sent(self, request_id):
        with self.lock:
            transaction = self.requests[request_id]
            if transaction["status"] != "reserved":
                return False
            transaction["status"] = "sent"
            return True

    def accept(self, request_id, *, now_s):
        with self.lock:
            transaction = self.requests[request_id]
            if transaction["status"] not in ("sent", "unknown"):
                return False
            request = transaction["request"]
            self.balance -= request.coins
            self.spent += request.coins
            self.version += 1
            transaction.update(status="accepted_pending_effect", accepted_at_s=now_s)
            return True

    def timeout(self, request_id):
        with self.lock:
            transaction = self.requests[request_id]
            if transaction["status"] == "sent":
                transaction["status"] = "unknown"

    def deny(self, request_id):
        with self.lock:
            transaction = self.requests[request_id]
            if transaction["status"] not in ("reserved", "sent", "unknown"):
                return False
            transaction["status"] = "denied"
            return True

    def effect(self, request_id, *, now_s, succeeded):
        with self.lock:
            transaction = self.requests[request_id]
            if transaction["status"] != "accepted_pending_effect":
                return False
            # Failure may be reported before the effect time (e.g. death).
            if succeeded:
                kind = transaction["request"].kind
                fact = "remote_heal_delay" if kind == "remote_heal" else "remote_ammo_delay"
                delay = rule_facts()[fact]["value"] if kind.startswith("remote") else 0.
                if now_s - transaction["accepted_at_s"] < delay - 1e-9:
                    return False
            transaction["status"] = "effect_confirmed" if succeeded else "failed_no_refund"
            # An already accepted purchase is never refunded by this method.
            return True
