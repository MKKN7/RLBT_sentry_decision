"""Checks of the physical fixture; real ROS/Nav2 is a separate integration run."""
import numpy as np
import pytest

from rl_bt.nav2_mujoco import MujocoNavigationFixture, occupancy_for_nav2, raster_rectangles
from rl_bt.ul_navigation import ULNavigation


def test_rectangles_preserve_holes():
    mask = np.array([[1, 1, 0, 1], [1, 1, 0, 1], [0, 1, 1, 0]], bool)
    rebuilt = np.zeros_like(mask)
    for x0, y0, x1, y1 in raster_rectangles(mask):
        assert not rebuilt[x0:x1, y0:y1].any()
        rebuilt[x0:x1, y0:y1] = True
    np.testing.assert_array_equal(rebuilt, mask)


def test_virtual_forbidden_supply_and_map_orientation():
    nav = ULNavigation()
    red = occupancy_for_nav2(nav, 0)
    blue = occupancy_for_nav2(nav, 1)
    assert red[105, 10] and not blue[105, 10]
    assert blue[15, 70] and not red[15, 70]
    assert red.shape == (120, 80)


def test_wrench_motion_watchdog_and_robot_parameters():
    sim = MujocoNavigationFixture(start_xy=(1.5, 4.))
    initial = sim.data.qpos[:2].copy()
    assert sim.model.body("chassis").mass[0] == 15.
    assert sim.model.nq > 7
    for _ in range(100):
        sim.set_command(.4, 0., 0.)
        sim.step(.01)
    assert sim.data.qpos[0] - initial[0] > .15
    assert abs(sim.data.qpos[1] - initial[1]) < .1
    assert sim.collision_steps == 0
    sim.step(1.)
    np.testing.assert_array_equal(sim.command, np.zeros(3))
    assert np.linalg.norm(sim.state()["local_velocity"][:2]) < .08
    with pytest.raises(ValueError, match="Finite"):
        sim.set_command(np.nan, 0., 0.)


def test_ground_course_obstacle_physically_blocks_drive():
    sim = MujocoNavigationFixture(start_xy=(1.5, 4.))
    for _ in range(200):
        sim.set_command(.4, 0., 0.)
        sim.step(.01)
    sim.step(1.)
    assert sim.collision_steps > 0
    assert sim.max_contact_force_n > 0.
    assert sim.data.qpos[0] < 2.2


def test_safety_interlock_rejects_new_motion_commands():
    sim = MujocoNavigationFixture(start_xy=(1.5, 4.))
    sim.set_safety_stop(True)
    sim.set_command(2., 1., 1.)
    np.testing.assert_array_equal(sim.command, np.zeros(3))
    sim.step(.1)
    assert sim.report()["safety_stopped"] is True
    sim.set_safety_stop(False)
    sim.set_command(.1, 0., 0.)
    assert sim.command[0] == .1
