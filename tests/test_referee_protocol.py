from dataclasses import asdict
import struct

import pytest

from rl_bt.referee_protocol import decode_payload


def decode(command, payload):
    return decode_payload(command, payload, byte_order="little")


def test_2027_hp_offsets_signed_difference_and_no_enemy_robot_hp():
    raw = struct.pack("<4Hh7H", 111, 65535, 333, 444, -123, 777,
                      1200, 3000, 1100, 2000, 590, 580)
    hp = decode(0x0003, raw)
    assert hp.ally_heavy_hp == 111 and hp.ally_sentry_hp == 777
    assert hp.ally_3_hp == 333 and hp.ally_4_hp == 444
    assert hp.damage_difference == -123
    assert hp.ally_control_degree == 590 and hp.enemy_control_degree == 580
    assert "reserved" not in asdict(hp)
    assert not any(key.startswith("enemy_") and "robot" in key for key in asdict(hp))
    assert set(key for key in asdict(hp) if key.startswith("enemy_")) == {
        "enemy_base_hp", "enemy_control_degree"}


def test_ul_zone_states_are_own_relative_with_no_hidden_capture_progress():
    second = (1 << 16) | (2 << 18) | (3 << 20) | (1 << 22) | (2 << 24)
    event = decode(0x0101, struct.pack("<II", 1, second))
    assert event.own_supply_active
    assert (event.r_highland.activation, event.r_highland.control) == (1, 2)
    assert (event.b_highland.activation, event.b_highland.control) == (3, 1)
    assert (event.center.activation, event.center.control) == (2, 0)
    with pytest.raises(ValueError, match="Undefined UL"):
        decode(0x0101, struct.pack("<II", 0, 3 << 26))


def test_rfid_supply_and_ul_cards_do_not_confuse_base_with_supply():
    rfid = decode(0x0209, struct.pack("<IH", (1 << 2) | (1 << 11) | (1 << 13), 65535))
    assert not rfid.own_supply_detected
    assert rfid.r_highland_detected and rfid.center_detected
    assert not rfid.b_highland_detected


def test_ally_positions_ignore_reserved_and_require_frame_registration():
    result = decode(0x020B, struct.pack("<10f", 0., 0., float("nan"), float("nan"),
                                     3., 4., 5., 6., float("nan"), float("nan")))
    assert result.heavy_xy == (0., 0.)
    assert result.infantry_3_xy == (3., 4.) and result.infantry_4_xy == (5., 6.)
    assert result.frame == "referee_field_frame_unregistered"
    with pytest.raises(ValueError, match="Nonfinite"):
        decode(0x020B, struct.pack("<10f", *([float("nan")] * 10)))


def test_status_and_allowance_have_new_2027_lengths():
    state = decode(0x0201, struct.pack("<BB5HfB", 7, 1, 390, 400, 50, 100, 100, 25., 5))
    assert state.hp == 390 and state.max_hp == 400
    assert state.gimbal_power and state.shooter_power and not state.chassis_power
    ammo = decode(0x0208, struct.pack("<4H", 750, 0, 1200, 0))
    assert ammo.allowance_17mm == 750 and ammo.gold == 1200


def test_game_status_keeps_ntp_timestamp_raw_and_result_uses_global_teams():
    status = decode(0x0001, struct.pack("<BHQ", 0x44, 299, 987654321))
    assert status.game_type == 4 and status.game_progress == 4
    assert status.stage_remaining_s == 299 and status.sync_timestamp_raw == 987654321
    assert decode(0x0002, b"\x02") == 2
    with pytest.raises(ValueError, match="winner"):
        decode(0x0002, b"\x03")


def test_malformed_payloads_and_radar_wireless_packets_are_rejected():
    with pytest.raises(ValueError, match="length"):
        decode(0x0003, b"\x00" * 32)
    with pytest.raises(ValueError, match="Unsupported"):
        decode(0x0A02, b"\x00" * 12)
    with pytest.raises(ValueError, match="byte_order"):
        decode_payload(0x0002, b"\x00", byte_order="native")
    with pytest.raises(TypeError, match="bytes"):
        decode(0x0002, [0])
    with pytest.raises(TypeError):
        decode_payload(0x0002, b"\x00")


def test_byte_order_is_explicit_and_portable():
    result = decode_payload(0x0208, struct.pack(">4H", 750, 0, 1000, 0), byte_order="big")
    assert result.allowance_17mm == 750 and result.gold == 1000
