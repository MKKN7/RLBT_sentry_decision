import struct

import pytest

from rl_bt.ul_observation import ULPublicObservationCache


def ingest(cache, command, payload, *, received=1., sampled=0.9, **kwargs):
    return cache.ingest(command, payload, byte_order="little", received_at_s=received,
                        sampled_at_s=sampled, **kwargs)


def test_stale_and_unknown_source_samples_never_expose_cached_actor_values():
    cache = ULPublicObservationCache()
    raw = struct.pack("<4H", 750, 0, 1234, 0)
    ingest(cache, 0x0208, raw, sampled=None)
    snap = cache.snapshot(now_s=1., max_sample_age_s={0x0208: .2})
    assert snap[0x0208].reason == "source_sample_time_unknown"
    assert snap[0x0208].receive_age_s == 0 and snap[0x0208].fields == {}
    ingest(cache, 0x0208, raw, received=1.1, sampled=.9)
    snap = cache.snapshot(now_s=1.11, max_sample_age_s={0x0208: .2})
    assert snap[0x0208].reason == "source_sample_stale" and snap[0x0208].fields == {}
    assert snap[0x0208].receive_age_s == pytest.approx(.01)
    assert snap[0x0208].sample_age_s == pytest.approx(.21)


def test_later_received_old_or_duplicate_samples_cannot_refresh_cache():
    cache = ULPublicObservationCache()
    raw = struct.pack("<4H", 740, 0, 1000, 0)
    assert ingest(cache, 0x0208, raw, sampled=.9, wire_sequence=254) == "stored"
    for received, sampled in ((1.1, .8), (1.2, .9), (1.3, None)):
        assert ingest(cache, 0x0208, raw, received=received, sampled=sampled,
                      wire_sequence=0) == "ignored_unknown_or_nonnewer_sample"
    snap = cache.snapshot(now_s=1.4, max_sample_age_s={0x0208: 1.})
    assert snap[0x0208].receive_age_s == pytest.approx(.4)
    assert snap[0x0208].wire_sequence == 254
    # A wrapped wire counter is valid metadata when the measured sample is new.
    assert ingest(cache, 0x0208, raw, received=1.5, sampled=1.4, wire_sequence=0) == "stored"


def test_ul_actor_allowlist_omits_enemy_hp_life_facilities_and_exact_progress():
    cache = ULPublicObservationCache()
    hp = struct.pack("<4Hh7H", 0, 0, 100, 0, -20, 400, 123, 456, 789, 999, 600, 550)
    ingest(cache, 0x0003, hp)
    ingest(cache, 0x0101, struct.pack("<II", 1, (1 << 16) | (1 << 18)))
    snap = cache.snapshot(now_s=1., max_sample_age_s={0x0003: .5, 0x0101: 2.})
    fields = {k: v for packet in snap.values() for k, v in packet.fields.items()}
    assert fields["ally_4_hp"] == 0  # no inferred dead/absent flag
    assert fields["ally_control_degree_raw"] == 600
    assert fields["r_highland_control"] == 1
    assert not any("enemy" in key and "hp" in key for key in fields)
    assert not any("progress" in key or "alive" in key or "base" in key for key in fields)


def test_positions_remain_masked_until_real_transport_and_frame_validation():
    cache = ULPublicObservationCache()
    ingest(cache, 0x020B, struct.pack("<10f", *([1.] * 10)))
    packet = cache.snapshot(now_s=1., max_sample_age_s={0x020B: 2.})[0x020B]
    assert not packet.valid and packet.fields == {}
    assert packet.reason == "ally_position_transport_and_frame_not_validated"


def test_missing_threshold_and_packets_have_explicit_masks():
    cache = ULPublicObservationCache()
    packet = cache.snapshot(now_s=0., max_sample_age_s={0x0201: .5})[0x0201]
    assert packet.reason == "never_received" and packet.sample_age_s is None
    assert cache.snapshot(now_s=0., max_sample_age_s={})[0x0201].reason == "sample_age_limit_not_configured"
    with pytest.raises(ValueError):
        cache.snapshot(now_s=0., max_sample_age_s={0x0201: 0.})


def test_profile_robot_source_and_monotonic_time_checks():
    cache = ULPublicObservationCache()
    with pytest.raises(ValueError, match="game_type"):
        ingest(cache, 0x0001, struct.pack("<BHQ", 0x41, 300, 0))
    with pytest.raises(ValueError, match="sentry"):
        ingest(cache, 0x0201, struct.pack("<BB5HfB", 3, 1, 100, 200, 0, 0, 100, 25., 7))
    with pytest.raises(ValueError, match="source"):
        ingest(cache, 0x0002, b"\x01", source="simulation_truth")
    with pytest.raises(ValueError, match="Sample time"):
        ingest(cache, 0x0002, b"\x01", sampled=1.1)
    ingest(cache, 0x0002, b"\x01")
    with pytest.raises(ValueError, match="monotonic"):
        ingest(cache, 0x0002, b"\x01", received=.9, sampled=.8)
    with pytest.raises(ValueError, match="follow"):
        cache.snapshot(now_s=.9, max_sample_age_s={})
