"""Selected RMU 2027 V1.0.0 payloads, after the upstream frame/CRC check.

This module never opens a serial port or treats decoded data as a fresh actor
observation. Transport availability, source/receive time, coordinate transforms
and profile-specific field permissions belong to the observation adapter.
Byte order is explicit: the PDF's offset tables do not specify payload byte order.
"""
from dataclasses import dataclass
import math
import struct


PROTOCOL_VERSION = "RMU_2027_V1.0.0_20261029"
SUPPORTED_PAYLOAD_LENGTHS = {
    0x0001: 11, 0x0002: 1, 0x0003: 24, 0x0101: 8,
    0x0201: 17, 0x0208: 8, 0x0209: 6, 0x020B: 40,
}


@dataclass(frozen=True)
class GameStatus:
    game_type: int
    game_progress: int
    stage_remaining_s: int
    sync_timestamp_raw: int  # valid only after the documented NTP connection


@dataclass(frozen=True)
class RobotHP:
    ally_heavy_hp: int
    ally_3_hp: int
    ally_4_hp: int
    damage_difference: int
    ally_sentry_hp: int
    outpost_r_hp: int
    ally_base_hp: int
    outpost_b_hp: int
    enemy_base_hp: int
    ally_control_degree: int
    enemy_control_degree: int


@dataclass(frozen=True)
class ULZoneStatus:
    activation: int  # 0 none, 1 own, 2 opponent, 3 both
    control: int  # 0 neutral, 1 own, 2 opponent; 3 is not defined


@dataclass(frozen=True)
class ULFieldEvents:
    own_supply_active: bool
    r_highland: ULZoneStatus
    b_highland: ULZoneStatus
    center: ULZoneStatus


@dataclass(frozen=True)
class RobotStatus:
    robot_id: int
    level: int
    hp: int
    max_hp: int
    barrel_cooling_per_s: int
    barrel_heat_limit: int
    chassis_power_limit_w: int
    bullet_speed_limit_m_s: float
    gimbal_power: bool
    chassis_power: bool
    shooter_power: bool


@dataclass(frozen=True)
class ProjectileAllowance:
    allowance_17mm: int
    allowance_42mm: int
    gold: int
    fortress_reserve: int


@dataclass(frozen=True)
class ULRfidStatus:
    own_supply_detected: bool
    r_highland_detected: bool
    b_highland_detected: bool
    center_detected: bool


@dataclass(frozen=True)
class AllyGroundPositions:
    heavy_xy: tuple[float, float]
    infantry_3_xy: tuple[float, float]
    infantry_4_xy: tuple[float, float]
    frame: str = "referee_field_frame_unregistered"


def decode_payload(command_id, payload, *, byte_order):
    """Decode one verified payload; do not pass an unchecked complete frame.

    ``byte_order`` must be established by the transport integration, e.g.
    ``little`` for a validated deployment. It is deliberately not inferred from
    host byte order. 0x0101 and 0x0209 expose only their UL-relevant fields.
    No radar-only enemy HP/position packets are accepted by this subset.
    """
    if type(command_id) is not int or command_id not in SUPPORTED_PAYLOAD_LENGTHS:
        raise ValueError("Unsupported regular-link payload command")
    if byte_order not in ("little", "big"):
        raise ValueError("Explicit validated byte_order must be little or big")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Payload must contain bytes")
    payload = bytes(payload)
    if len(payload) != SUPPORTED_PAYLOAD_LENGTHS[command_id]:
        raise ValueError("Payload length does not match the 2027 protocol")
    prefix = "<" if byte_order == "little" else ">"
    def unpack(fmt):
        return struct.unpack(prefix + fmt, payload)
    if command_id == 0x0001:
        packed, remaining, timestamp = unpack("BHQ")
        return GameStatus(packed & 15, packed >> 4, remaining, timestamp)
    if command_id == 0x0002:
        winner, = unpack("B")
        if winner not in (0, 1, 2):
            raise ValueError("Undefined game-result winner")
        return winner
    if command_id == 0x0003:
        values = unpack("4Hh7H")
        # Offset 2 is reserved. The damage difference at offset 8 is signed.
        return RobotHP(values[0], *values[2:])
    if command_id == 0x0101:
        first, second = unpack("II")
        zones = []
        for bit in (16, 20, 24):
            activation = (second >> bit) & 3
            control = (second >> (bit + 2)) & 3
            if control == 3:
                raise ValueError("Undefined UL zone-control state")
            zones.append(ULZoneStatus(activation, control))
        return ULFieldEvents(bool(first & 1), *zones)
    if command_id == 0x0201:
        values = unpack("BB5HfB")
        if not math.isfinite(values[-2]):
            raise ValueError("Nonfinite robot speed limit")
        flags = values[-1]
        return RobotStatus(*values[:-1], bool(flags & 1), bool(flags & 2), bool(flags & 4))
    if command_id == 0x0208:
        return ProjectileAllowance(*unpack("4H"))
    if command_id == 0x0209:
        first, _ = unpack("IH")
        return ULRfidStatus(*(bool(first & (1 << bit)) for bit in (0, 11, 12, 13)))
    values = unpack("10f")
    named = (values[0:2], values[4:6], values[6:8])
    if not all(math.isfinite(x) for pair in named for x in pair):
        raise ValueError("Nonfinite named ally position")
    # Reserved slots never become engineer/sentry/enemy coordinates.
    # (0, 0) is not a documented invalid sentinel and is not silently discarded.
    return AllyGroundPositions(*named)
