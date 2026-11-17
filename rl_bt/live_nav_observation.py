"""Policy-visible features from live Point-LIO/BT/observed support only.

No simulator position, STL surface, replay motion or hidden opponent state is
read here. Mask computation is outside the policy and never marks unknown free.
Candidate admissibility here is only an observed straight-corridor prefilter;
the ROS adapter must also request and verify the actual Nav2 path at admission.
"""
from __future__ import annotations

import math
import numpy as np

from rl_bt.support_costmap import path_support_check

LIVE_NAV_SCENARIO_ID = "uc_original_live_pointlio_nav2_mppi_task_v1"
LIVE_NAV_CONTRACT_REVISION = 2
RADIUS_M = .411


def _age(raw, key):
    value = raw.get(key)
    return float(value) if value is not None and math.isfinite(float(value)) else 1000.


def build_live_observation(raw, support_grid):
    """Return finite arrays, 13 indexed target poses, and auditable BT checks."""
    pose = np.asarray(raw.get("pose_xy_m", [0., 0.]), float)
    goal = np.asarray(raw["task_goal_xy_m"], float)
    if pose.shape != (2,) or goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("Finite task XY and two-dimensional estimated pose required")
    pose_valid = np.isfinite(pose).all() and _age(raw, "pose_age_s") <= .75
    if not np.isfinite(pose).all():
        pose = np.zeros(2)
    elapsed = max(0., float(raw.get("task_elapsed_s", 0.)))
    deadline = max(.5, float(raw.get("task_timeout_s", 20.)))
    support_age = _age(raw, "support_age_s")
    fresh = (support_grid is not None and support_age <= 2.5 and
        _age(raw, "tf_age_s") <= .75 and _age(raw, "obstacle_age_s") <= 2.5)
    healthy = bool(raw.get("health", False) and pose_valid and fresh and
                   not raw.get("safety_latched", False))
    current_check = (path_support_check(support_grid, [pose], radius_m=RADIUS_M)
        if pose_valid and support_grid is not None else {"accepted": False, "reason": "missing_pose_or_support"})
    admissible = healthy and current_check["accepted"]
    active_raw = raw.get("active_goal_xy_m")
    active = None if active_raw is None else np.asarray(active_raw, float)
    if active is not None and (active.shape != (2,) or not np.isfinite(active).all()):
        raise ValueError("Invalid active goal XY")
    status = str(raw.get("executor_state", "idle"))
    pending = status in ("planning", "awaiting_acceptance", "canceling", "uncertain")
    distance = float(np.linalg.norm(goal-pose))
    state = np.array([*(goal-pose), distance/5., max(0., 1.-elapsed/deadline),
        float(raw.get("terminal_status") == "succeeded"), 1./5.,
        float(active is not None), min(float(raw.get("active_age_s",0.))/10.,1.),
        min(float(raw.get("stall_age_s",0.))/3.,1.), float(admissible),
        min(support_age/3.,1.), float(raw.get("last_veto",False)),
        float(raw.get("last_arrival",False)),float(raw.get("last_recovery",False)),
        *(np.zeros(2) if active is None else active-pose),
        (0. if active is None else np.linalg.norm(active-pose)/5.),
        float(pending), 0., float(pose_valid),
        float(raw.get("last_progress_m",0.))/.15,0.,0.,
        min(float(raw.get("time_since_goal_switch_s",0.))/5.,1.)],dtype=np.float32)
    # Engineering task suggestions are generated geometrically; they do not
    # assume support until each full disk sweep passes the actual observed map.
    proposed = [goal]
    delta = goal-pose
    length = float(np.linalg.norm(delta))
    if length > 1e-8:
        for travel in (.15,.25,.3):
            proposed.append(pose+delta/length*min(travel,length))
    proposed.extend(pose+np.array(offset) for offset in
        ((0.,.15),(0.,-.15),(.15,0.),(-.15,0.),(.15,.15),(-.15,.15),(.15,-.15),(-.15,-.15)))
    if active is not None:
        proposed.append(active)
    points = []
    for point in proposed:
        if not any(np.linalg.norm(point-other)<1e-5 for other in points):
            points.append(np.asarray(point,float))
    points=points[:13]
    goals = [None]*13
    candidates=np.zeros((16,12),dtype=np.float32)
    mask=np.zeros(16,dtype=bool)
    checks=[]
    for index,point in enumerate(points):
        goals[index]=point.tolist()
        check=(path_support_check(support_grid,[pose,point],radius_m=RADIUS_M)
            if support_grid is not None and pose_valid else {"accepted":False,"reason":"missing_pose_or_support"})
        same=active is not None and np.linalg.norm(point-active)<1e-5
        # The ROS adapter serializes goals. An unrelated replacement is not
        # executable until the active task reaches an actual terminal result.
        # Keep the mask consistent with that existing capability restriction.
        mask[index]=admissible and check["accepted"] and not pending and (active is None or same)
        candidates[index]=[*(point-pose),np.linalg.norm(point-pose),np.linalg.norm(point-goal),
            float(np.linalg.norm(point-goal)<1e-5),1.,0.,0.,0.,float(same),min(support_age/3.,1.),float(mask[index])]
        checks.append({"candidate_id":index,"goal_xy_m":point.tolist(),**check})
    candidates[13,6]=1.
    candidates[14,7]=1.
    candidates[15,8]=1.
    mask[13]=active is not None and status not in ("canceling","uncertain")
    mask[14]=active is not None and admissible and status not in ("canceling","uncertain")
    mask[15]=True
    if active is not None:
        candidates[14,:4]=[*(active-pose),np.linalg.norm(active-pose),np.linalg.norm(active-goal)]
        candidates[14,4]=float(np.linalg.norm(active-goal)<1e-5)
        candidates[14,9]=1.
    candidates[13:,11]=mask[13:]
    state[18]=float(not mask[:13].any() and active is None)
    state[21]=mask.sum()/16.
    observation={"state":np.clip(state,-10.,10.),"candidates":np.clip(candidates,-10.,10.),"mask":mask}
    diagnostics={"current_footprint_check":current_check,"candidate_checks":checks,
        "healthy_for_admission":admissible,"unknown_filled":False,
        "GT_used_as_policy_input":False,"radius_m":RADIUS_M,
        "observation_schema":LIVE_NAV_SCENARIO_ID}
    diagnostics["contract_revision"]=LIVE_NAV_CONTRACT_REVISION
    diagnostics["goal_replacement_requires_terminal"]=True
    return observation,goals,diagnostics


def live_nav_bt_baseline(observation):
    """Observation-only commissioning policy; not labeled as RL training."""
    mask=observation["mask"]
    if mask[14]: return 14
    available=np.flatnonzero(mask[:13])
    if len(available):
        c=observation["candidates"]
        return int(min(available,key=lambda i:(c[i,3],c[i,2])))
    return 15
