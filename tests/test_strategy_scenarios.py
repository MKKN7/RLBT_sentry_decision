"""Role-adapter contracts with small mocks; no simulated match claims."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rl_bt.alpharm_task_env import AlphaTaskEnv
from rl_bt.strategy_scenarios import (DEFAULT_CONFIG, RAW_CONTINUOUS_BOUND, RoleBalancedTaskEnv,
    SCENARIO_ID, _inverse_continuous)


def config_path(tmp_path, **updates):
    config=json.loads(DEFAULT_CONFIG.read_text())
    config.update(updates)
    path=tmp_path/"scene.json"
    path.write_text(json.dumps(config))
    return path


class BaseActor:
    def __init__(self):
        self.calls=0

    def run(self,names,feeds):
        self.calls+=1
        raw=np.full((6,14),.1,np.float32)
        return raw,np.asarray(feeds["hidden"])+1


@pytest.fixture
def initialized(monkeypatch):
    def fake_init(self,runtime,seed,**kwargs):
        self.runtime=runtime
        self.team=kwargs.get("side") or 0
        self.style=kwargs.get("opponent_style") or "capture"
        self.obs=np.zeros((6,140),np.float32)
        self.obs[:,7]=1.
        self.reset()
    def fake_reset(self):
        self.reset_calls=getattr(self,"reset_calls",0)+1
        self.cached={"state":np.zeros(148,np.float32),"candidates":np.zeros((16,16),np.float32),"mask":np.ones(16,bool)}
        return self.cached
    monkeypatch.setattr(AlphaTaskEnv,"__init__",fake_init)
    monkeypatch.setattr(AlphaTaskEnv,"reset",fake_reset)
    monkeypatch.setattr(AlphaTaskEnv,"state_dict",lambda self:{
        "alive":np.ones(6,bool),"target":np.zeros(6,np.int32),"invuln":np.zeros(6),
        "overheated":np.zeros(6,bool),"heat":np.zeros(6)})
    monkeypatch.setattr(AlphaTaskEnv,"fixed_combat_action",lambda self,i,tm,fm:np.array([0.,0.,.35,0.,0.,0.],np.float32))
    monkeypatch.setattr(AlphaTaskEnv,"navigation_command",lambda self,i,candidate:np.array([.3,-.2],np.float32))
    return SimpleNamespace(actor=BaseActor(),fixture={"config":{"lock_range":12.,"heat_max":100.}})


def test_inverse_actor_decoder_roundtrips_representable_commands_and_reports_bound():
    command=np.array([-.9,.3,.35,0.])
    raw,error=_inverse_continuous(command)
    np.testing.assert_allclose(np.tanh(3*np.tanh(raw/3)),command,atol=2e-7)
    assert error==0.
    raw,error=_inverse_continuous(np.array([-1.,1.,0.,0.]))
    assert np.isfinite(raw).all()
    assert error==pytest.approx(1-RAW_CONTINUOUS_BOUND+1e-7)
    assert error<.005


def test_runtime_facade_is_per_environment_and_does_not_mutate_shared_actor(initialized):
    original=initialized.actor
    first=RoleBalancedTaskEnv(initialized,opponent_style="capture",side=0)
    second=RoleBalancedTaskEnv(initialized,opponent_style="pretrained",side=1)
    assert initialized.actor is original
    assert first.runtime.actor is not second.runtime.actor
    assert first.runtime.actor.owner is first and second.runtime.actor.owner is second
    assert first.reset_calls==second.reset_calls==1
    assert first.config_sha256==hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()


@pytest.mark.parametrize("side",[0,1])
def test_anchor_roles_and_observed_hp_retreat_are_mirrored_and_frontline_preserved(initialized,monkeypatch,side):
    monkeypatch.setattr(AlphaTaskEnv,"scripted_candidate",lambda self,robot:14)
    env=RoleBalancedTaskEnv(initialized,side=side,support_mode="anchors")
    own=side*3
    assert [env._own_support_candidate(own+i) for i in (0,1)]==[5,7]
    assert [env.scripted_candidate(i) for i in range(6)]==[5,7,14,5,7,14]
    env.obs[own,7]=.3
    assert env._own_support_candidate(own)==2
    assert env.scripted_candidate(own)==2
    with pytest.raises(ValueError):
        env._own_support_candidate(own+2)


def test_balanced_own_support_never_inherits_pretrained_enemy_behavior(initialized,monkeypatch):
    monkeypatch.setattr(AlphaTaskEnv,"scripted_candidate",lambda self,robot:14)
    env=RoleBalancedTaskEnv(initialized,opponent_style="pretrained",support_mode="balanced")
    assert env._own_support_candidate(0)==1 and env._own_support_candidate(1)==3
    assert env.scripted_candidate(4)==14
    env.obs[1,7]=.2
    assert env._own_support_candidate(1)==2


def test_scene_configured_retreat_threshold_applies_to_both_mirrored_supports(initialized,tmp_path):
    path=config_path(tmp_path,support_retreat_hp_fraction=.7)
    env=RoleBalancedTaskEnv(initialized,config_path=path,side=0,support_mode="anchors")
    env.obs[:,7]=.6
    assert env._own_support_candidate(0)==2
    assert env.scripted_candidate(3)==2
    assert env.scene_metadata()["support_retreat_hp_threshold"]==.7


def test_scripted_scene_skips_base_actor_and_returns_fixed_support_decoding(initialized):
    env=RoleBalancedTaskEnv(initialized,opponent_style="capture")
    hidden=np.ones((6,8),np.float32)
    raw,next_hidden=env.runtime.actor.run(None,{"obs":env.obs,"hidden":hidden})
    assert initialized.actor.calls==0
    np.testing.assert_array_equal(hidden,next_hidden)
    decoded=np.tanh(3*np.tanh(raw[:,:4]/3))
    np.testing.assert_allclose(decoded[:2],np.tile([.3,-.2,.35,0.],(2,1)),atol=2e-7)
    assert np.argmax(raw[:2,8:12],axis=-1).tolist()==[0,0]
    assert np.argmax(raw[:2,12:14],axis=-1).tolist()==[0,0]
    metadata=env.scene_metadata()
    assert metadata["base_actor_calls_skipped"]==1
    assert metadata["support_candidate_counts"]==[{5:1},{7:1}]


def test_pretrained_enemy_actions_hidden_preserved_and_only_own_support_overridden(initialized):
    env=RoleBalancedTaskEnv(initialized,opponent_style="pretrained",side=1)
    hidden=np.ones((6,8),np.float32)
    raw,next_hidden=env.runtime.actor.run(None,{"obs":env.obs,"hidden":hidden})
    assert initialized.actor.calls==1
    np.testing.assert_array_equal(next_hidden,hidden+1)
    np.testing.assert_array_equal(raw[[0,1,2,5]],np.full((4,14),.1,np.float32))
    assert not np.all(raw[3:5]==.1)
    assert env.scene_metadata()["enemy_uses_pretrained_actor"]


def test_inherited_step_reward_observation_done_preserved_and_metadata_versioned(initialized,monkeypatch):
    env=RoleBalancedTaskEnv(initialized)
    def step(self,action):
        assert action==7
        return self.cached,3.25,True,{"episode":{"scenario_id":"old","win":True}}
    monkeypatch.setattr(AlphaTaskEnv,"step",step)
    observation,reward,done,info=env.step(7)
    assert observation is env.cached and reward==3.25 and done is True
    assert info["episode"]["scenario_id"]==SCENARIO_ID
    assert info["episode"]["strategy_scene"]["true_done_preserved"]
    env._base_actor_calls=12
    env.reset()
    assert env._base_actor_calls==0 and env.reset_calls==2


@pytest.mark.parametrize("change",[
    {"scenario_id":"other"},{"observation_semantics":"other"},
    {"real_nav2_in_loop":True},{"official_UL_UC_training":True},
    {"anchor_candidates":[1,3]},{"support_retreat_hp_fraction":-1},
])
def test_invalid_scene_contract_rejected_before_old_environment_initializes(tmp_path,change):
    with pytest.raises(ValueError):
        RoleBalancedTaskEnv(SimpleNamespace(),config_path=config_path(tmp_path,**change))
