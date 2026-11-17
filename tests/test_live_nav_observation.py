import numpy as np

from rl_bt.live_nav_observation import build_live_observation, live_nav_bt_baseline
from rl_bt.voxel_terrain import TerrainGrid


def grid(value=0):
    values=np.full((30,30),value,np.int8)
    return TerrainGrid('odom',5.,np.array([-1.5,-1.5]),.1,values,
        np.empty((0,0)),np.empty((0,0)),np.empty((0,3)),{})


def raw(**extra):
    return dict(now_s=5.,pose_xy_m=[0.,0.],task_goal_xy_m=[0.,.4],
        health=True,pose_age_s=0.,tf_age_s=0.,obstacle_age_s=0.,support_age_s=0.,**extra)


def test_mask_is_observed_support_and_never_unknown_fill():
    for value in (-1,100):
        obs,goals,diag=build_live_observation(raw(),grid(value))
        assert not obs['mask'][:13].any() and obs['mask'][15]
        assert not diag['unknown_filled']
    obs,goals,diag=build_live_observation(raw(),grid())
    assert obs['mask'][0] and live_nav_bt_baseline(obs)==0
    assert goals[0]==[0.,.4]


def test_stale_or_unsupported_current_footprint_cannot_dispatch():
    r=raw();r['support_age_s']=2.6
    obs,_,_=build_live_observation(r,grid())
    assert not obs['mask'][:13].any()
    g=grid();g.occupancy_xy[14,15]=-1
    obs,_,diag=build_live_observation(raw(),g)
    assert not diag['current_footprint_check']['accepted']
    assert not obs['mask'][:13].any()


def test_active_continue_not_new_goal_during_async_admission():
    obs,_,_=build_live_observation(raw(active_goal_xy_m=[0.,.4],executor_state='awaiting_acceptance'),grid())
    assert not obs['mask'][:13].any() and obs['mask'][14]
    assert live_nav_bt_baseline(obs)==14
    assert obs['candidates'][14,4]==1


def test_no_truth_dependency_and_nonfinite_pose_has_hold_fallback():
    r=raw();r['pose_xy_m']=[float('nan'),0.]
    obs,_,diag=build_live_observation(r,grid())
    assert np.isfinite(obs['state']).all() and obs['mask'][15]
    assert not obs['mask'][:13].any() and not diag['GT_used_as_policy_input']


def test_active_mask_matches_serialized_goal_adapter_without_hiding_hold():
    obs,goals,diag=build_live_observation(raw(active_goal_xy_m=[0.,.4],executor_state='active'),grid())
    assert obs['mask'][0] and obs['mask'][14] and obs['mask'][13] and obs['mask'][15]
    for index in np.flatnonzero(obs['mask'][:13]):
        np.testing.assert_allclose(goals[index],[0.,.4])
    assert diag['goal_replacement_requires_terminal']


def test_pending_planner_can_continue_without_forced_hold_cancellation():
    obs,_,_=build_live_observation(raw(active_goal_xy_m=[0.,.4],executor_state='planning'),grid())
    assert obs['mask'][14] and obs['mask'][15] and not obs['mask'][:13].any()
    assert live_nav_bt_baseline(obs)==14
