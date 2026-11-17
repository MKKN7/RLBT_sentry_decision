"""CPU MuJoCo execution fixture for real Nav2/MPPI integration checks.

The robot and drive limits come from the frozen navigation simulation. The
scene is an extruded UL ground obstacle raster, not the CAD contact geometry.
The fixture supplies ground-truth odometry explicitly; it does not test SLAM.
ROS imports live in the runner so the physics can be checked without ROS.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .ul_navigation import ROOT, ULNavigation


def raster_rectangles(mask):
    """Merge equal contiguous occupied runs across rows, without filling holes."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Two-dimensional occupancy required")
    active, result = {}, []
    for ix in range(mask.shape[0] + 1):
        runs = []
        if ix < mask.shape[0]:
            changes = np.diff(np.r_[False, mask[ix], False].astype(np.int8))
            runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        current = set(runs)
        for run in tuple(active):
            if run not in current:
                result.append((active.pop(run), run[0], ix, run[1]))
        for run in runs:
            active.setdefault(run, ix)
    return result


def occupancy_for_nav2(nav, team):
    """Return [x,y] raw occupancy; inflation is performed once by Nav2."""
    if team not in (0, 1) or isinstance(team, bool):
        raise ValueError("Explicit red=0 / blue=1 team required")
    occupied = ~nav.supported_cells | nav.obstacle_cells
    ix, iy = np.indices(nav.shape)
    xy = nav.origin_xy_m + (np.stack((ix, iy), axis=-1) + .5) * nav.resolution_m
    lower, upper = (([9.5, 0.], [12., 2.]) if team == 0 else ([0., 6.], [2.5, 8.]))
    # This is a virtual forbidden area, not an added physical wall.
    forbidden = np.all(xy >= lower, axis=-1) & np.all(xy <= upper, axis=-1)
    return occupied | forbidden


class MujocoNavigationFixture:
    """Omni wrench drive, finite commands and simulation-time watchdog.

    No velocity or pose teleportation is used during execution. Reset is limited
    to constructing a new fixture. Sensor generation and Point-LIO are omitted.
    """
    def __init__(self, *, start_xy=(1.5, 7.), yaw=0., team=0, nav=None):
        self.nav = ULNavigation() if nav is None else nav
        if not self.nav.is_free(start_xy, team):
            raise ValueError("Spawn must lie in a valid ground-course cell")
        self.team = team
        snapshot = ROOT / "assets/navigation_reference"
        sys.path.insert(0, str(snapshot / "source"))
        from mid360_mujoco.config import load
        from mid360_mujoco.model import build, robot_config
        self.config = load(snapshot / "configs/simulation.yaml")
        self.robot_config = robot_config(snapshot / "configs/robot.yaml")
        tree = ET.fromstring(build(self.config, self.robot_config, field=None))
        world = tree.find("worldbody")
        # Only replace the upstream test scene, preserving every robot body,
        # geom, inertia, suspension, and actuator parameter.
        for geom in list(world.findall("geom")):
            world.remove(geom)
        ET.SubElement(world, "geom", name="fixture_ground", type="plane", size="15 15 .1", rgba=".4 .45 .45 1")
        raw_occupied = ~self.nav.supported_cells | self.nav.obstacle_cells
        self.obstacle_rectangles = raster_rectangles(raw_occupied)
        for number, (x0, y0, x1, y1) in enumerate(self.obstacle_rectangles):
            lo = self.nav.origin_xy_m + np.array([x0, y0]) * self.nav.resolution_m
            hi = self.nav.origin_xy_m + np.array([x1, y1]) * self.nav.resolution_m
            centre, half = (lo + hi) / 2, (hi - lo) / 2
            ET.SubElement(world, "geom", name=f"fixture_obstacle_{number}", type="box",
                pos=f"{centre[0]} {centre[1]} .35", size=f"{half[0]} {half[1]} .35", rgba=".6 .6 .65 1")
        chassis = world.find("body[@name='chassis']")
        chassis.set("pos", f"{start_xy[0]} {start_xy[1]} .03")
        chassis.set("euler", f"0 0 {float(yaw)}")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body("chassis").id
        self.dt = float(self.model.opt.timestep)
        self.safety_stopped = False
        self.command = np.zeros(3)
        self.target = np.zeros(3)
        self.command_at_s = -np.inf
        self.collision_steps = 0
        self.max_contact_force_n = 0.
        self.max_abs_roll_pitch_rad = 0.
        self.command_count = 0
        self.max_received_speed_m_s = 0.
        self.path = []
        mujoco.mj_forward(self.model, self.data)
        # Let the original suspension settle before exposing clock/TF.
        self.step(2.)
        self.path.clear()
        self.collision_steps = 0

    def set_command(self, vx, vy, wz):
        value = np.array([vx, vy, wz], dtype=float)
        if not np.isfinite(value).all():
            raise ValueError("Finite velocity command required")
        drive = self.config["drive"]
        speed = np.linalg.norm(value[:2])
        self.max_received_speed_m_s = max(self.max_received_speed_m_s, float(speed))
        if speed > drive["max_speed"]:
            value[:2] *= drive["max_speed"] / speed
        value[2] = np.clip(value[2], -drive["max_yaw_rate"], drive["max_yaw_rate"])
        self.command[:] = 0. if self.safety_stopped else value
        self.command_at_s = self.data.time
        self.command_count += 1

    def set_safety_stop(self, active):
        if type(active) is not bool:
            raise ValueError("Explicit Boolean safety interlock required")
        self.safety_stopped = active
        if active:
            self.command[:] = 0.

    def step(self, duration_s):
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Positive finite step duration required")
        count = round(duration_s / self.dt)
        if count < 1 or abs(count * self.dt - duration_s) > 1e-8:
            raise ValueError("Step duration must be an integer physics timestep")
        drive = self.config["drive"]
        for _ in range(count):
            if self.data.time - self.command_at_s > drive["command_timeout"]:
                self.command[:] = 0.
            delta = self.command - self.target
            magnitude = np.linalg.norm(delta[:2])
            self.target[:2] += delta[:2] * min(1., drive["linear_accel"] * self.dt / max(magnitude, 1e-12))
            self.target[2] += np.clip(delta[2], -drive["angular_accel"] * self.dt, drive["angular_accel"] * self.dt)
            local = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, local, 1)
            rotation = self.data.xmat[self.base].reshape(3, 3)
            force = np.r_[(self.target[:2] - local[3:5]) * drive["velocity_gain"] * 15., 0.]
            norm = np.linalg.norm(force)
            if norm > drive["force_limit"]:
                force *= drive["force_limit"] / norm
            torque = np.array([0., 0., np.clip(drive["yaw_gain"] * (self.target[2] - local[2]), -drive["torque_limit"], drive["torque_limit"])])
            grounded = any(self.model.geom_group[self.data.contact[i].geom1] == 2 or self.model.geom_group[self.data.contact[i].geom2] == 2 for i in range(self.data.ncon))
            self.data.xfrc_applied[self.base, :3] = rotation @ force if grounded else 0.
            self.data.xfrc_applied[self.base, 3:] = rotation @ torque if grounded else 0.
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("MuJoCo fixture diverged")
            collision = False
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                names = (self.model.geom(int(contact.geom1)).name or "", self.model.geom(int(contact.geom2)).name or "")
                if any(name.startswith("fixture_obstacle_") for name in names):
                    collision = True
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
                    self.max_contact_force_n = max(self.max_contact_force_n, float(np.linalg.norm(wrench[:3])))
            self.collision_steps += int(collision)
        self.path.append([float(self.data.time), *self.data.qpos[:3].tolist()])
        rotation = self.data.xmat[self.base].reshape(3, 3)
        self.max_abs_roll_pitch_rad = max(self.max_abs_roll_pitch_rad, float(np.arccos(np.clip(rotation[2, 2], -1., 1.))))

    def state(self):
        local = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, local, 1)
        q = self.data.qpos[3:7]
        return {"time_s": float(self.data.time), "position": self.data.qpos[:3].tolist(), "quaternion_wxyz": q.tolist(),
                "local_velocity": local[3:].tolist(), "local_angular_velocity": local[:3].tolist()}

    def report(self):
        return {"fixture": "ul_ground_raster_mujoco_v1", "robot_model": "frozen_navigation_reference_unmodified_parameters",
                "odometry_source": "simulation_ground_truth_execution_validation_only", "pointlio_validated": False,
                "cad_contact_validated": False, "tunnel_validated": False,
                "time_s": float(self.data.time), "command_count": self.command_count, "safety_stopped": self.safety_stopped,
                "collision_physics_steps": self.collision_steps, "max_contact_force_n": self.max_contact_force_n,
                "max_abs_tilt_rad": self.max_abs_roll_pitch_rad, "position": self.data.qpos[:3].tolist(),
                "physics_timestep_s": self.dt, "obstacle_rectangles": len(self.obstacle_rectangles),
                "max_received_speed_m_s": self.max_received_speed_m_s, "trajectory": self.path}
