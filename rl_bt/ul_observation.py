"""Strict timestamped UL public-referee observation projection.

The cache is an offline adapter, not a serial/frame driver. Sample times must be
reconciled into the caller's monotonic clock by the transport layer. Missing
source times remain unknown; receipt alone cannot prove measurement freshness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from rl_bt.referee_protocol import decode_payload, SUPPORTED_PAYLOAD_LENGTHS


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Finite nonnegative {name} required")
    return float(value)


@dataclass(frozen=True)
class PublicPacket:
    valid: bool
    reason: str
    receive_age_s: float | None
    sample_age_s: float | None
    wire_sequence: int | None
    fields: dict


@dataclass(frozen=True)
class _Stored:
    value: object
    received_at_s: float
    sampled_at_s: float | None
    wire_sequence: int | None


def _project(command, value):
    """UL allowlist: no enemy robots, hidden capture progress or inferred life."""
    if command == 0x0001:
        return {"game_type": value.game_type, "game_progress": value.game_progress,
                "stage_remaining_s": value.stage_remaining_s}
    if command == 0x0002:
        return {"match_winner_red_blue": value}
    if command == 0x0003:
        return {"ally_heavy_hp": value.ally_heavy_hp, "ally_3_hp": value.ally_3_hp,
                "ally_4_hp": value.ally_4_hp, "ally_sentry_hp": value.ally_sentry_hp,
                "team_damage_difference": value.damage_difference,
                "ally_control_degree_raw": value.ally_control_degree,
                "enemy_control_degree_raw": value.enemy_control_degree}
    if command == 0x0101:
        fields = {"own_supply_active": value.own_supply_active}
        for zone in ("r_highland", "b_highland", "center"):
            status = getattr(value, zone)
            fields[f"{zone}_activation"] = status.activation
            fields[f"{zone}_control"] = status.control
        return fields
    if command == 0x0201:
        return {"own_robot_id": value.robot_id, "own_hp": value.hp,
                "own_max_hp": value.max_hp, "own_chassis_power_output": value.chassis_power}
    if command == 0x0208:
        return {"own_allowance_17mm": value.allowance_17mm, "gold_raw": value.gold}
    if command == 0x0209:
        return {"own_supply_rfid": value.own_supply_detected,
                "r_highland_rfid": value.r_highland_detected,
                "b_highland_rfid": value.b_highland_detected,
                "center_rfid": value.center_detected}
    return {}  # 0x020B is blocked until live coverage and frame registration.


class ULPublicObservationCache:
    """Per-match cache preserving validity and rejecting source-time regression.

    Wire sequence is metadata only: it is an 8-bit global frame counter, not a
    per-command monotonic timestamp. A later received old sample cannot overwrite
    a newer sample. No implicit age threshold or transport-delay bound is added.
    Create a new instance at the match/session boundary. Packets have independent
    validity; the caller still requires a valid UL game-status before acting.
    """
    def __init__(self):
        self._packets: dict[int, _Stored] = {}
        self._now = 0.

    def ingest(self, command_id, verified_payload, *, byte_order, received_at_s,
               sampled_at_s, source="referee_regular", wire_sequence=None):
        if source != "referee_regular":
            raise ValueError("Only verified regular-link source is permitted")
        received = _time(received_at_s, "receive time")
        sampled = None if sampled_at_s is None else _time(sampled_at_s, "sample time")
        if sampled is not None and sampled > received:
            raise ValueError("Sample time must be reconciled before receive time")
        if wire_sequence is not None and (type(wire_sequence) is not int or not 0 <= wire_sequence <= 255):
            raise ValueError("Wire sequence must be an unsigned 8-bit counter")
        if received < self._now:
            raise ValueError("Receive event time must be monotonic")
        value = decode_payload(command_id, verified_payload, byte_order=byte_order)
        if command_id == 0x0001 and value.game_type != 4:
            raise ValueError("UL 3v3 observations require game_type 4")
        if command_id == 0x0201 and value.robot_id not in (7, 107):
            raise ValueError("Own robot-status payload must belong to a sentry")
        self._now = received
        previous = self._packets.get(command_id)
        if previous is not None and previous.sampled_at_s is not None:
            if sampled is None or sampled <= previous.sampled_at_s:
                return "ignored_unknown_or_nonnewer_sample"
        self._packets[command_id] = _Stored(value, received, sampled, wire_sequence)
        return "stored"

    def snapshot(self, *, now_s, max_sample_age_s):
        """Return separate packet masks; invalid packets expose no cached values.

        ``max_sample_age_s`` maps command IDs to caller-selected positive age
        limits. A missing threshold disables that packet. These limits are
        engineering settings and must not be presented as official rule values.
        """
        now = _time(now_s, "snapshot time")
        if now < self._now:
            raise ValueError("Snapshot time must follow received events")
        for command, age in max_sample_age_s.items():
            if command not in SUPPORTED_PAYLOAD_LENGTHS or _time(age, "sample age limit") <= 0:
                raise ValueError("Supported command and positive sample-age limit required")
        self._now = now
        result = {}
        for command in SUPPORTED_PAYLOAD_LENGTHS:
            stored = self._packets.get(command)
            receive_age = None if stored is None else now - stored.received_at_s
            sample_age = None if stored is None or stored.sampled_at_s is None else now - stored.sampled_at_s
            if command == 0x020B:
                reason = "ally_position_transport_and_frame_not_validated"
            elif command not in max_sample_age_s:
                reason = "sample_age_limit_not_configured"
            elif stored is None:
                reason = "never_received"
            elif sample_age is None:
                reason = "source_sample_time_unknown"
            elif sample_age > max_sample_age_s[command]:
                reason = "source_sample_stale"
            else:
                reason = "valid"
            valid = reason == "valid"
            result[command] = PublicPacket(valid, reason, receive_age, sample_age,
                None if stored is None else stored.wire_sequence,
                _project(command, stored.value) if valid else {})
        return result
