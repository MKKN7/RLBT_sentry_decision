"""Fixed-controller audit semantics; fixture success is not a match result."""
import numpy as np
import pytest

from scripts.audit_temporal_policy_leverage import fixed_control, run_control_trial, audit


@pytest.mark.parametrize("policy", ["continue_only", "fixed_center"])
def test_controller_uses_only_mask_and_always_chooses_legal_fallback(policy):
    mask = np.zeros(16, bool)
    mask[[0,1,8]] = True
    assert fixed_control({"mask":mask}, policy) == (0 if policy=="continue_only" else 1)
    mask[0]=mask[1]=False
    assert fixed_control({"mask":mask}, policy) == 8
    mask[0]=True
    assert fixed_control({"mask":mask}, policy) == 0


@pytest.mark.parametrize("mask", [np.ones(16,np.float32),np.ones(15,bool),np.zeros(16,bool)])
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        fixed_control({"mask":mask}, "continue_only")


class FixtureEnvironment:
    def __init__(self, runtime, seed, **kwargs):
        self.cached={"mask":np.ones(16,bool)}
        self.learner=2
        self.dropout=np.zeros(3,bool)
        self.obs=np.zeros((6,140),np.float32)
        self.obs[2,[7,11,21]]=[.8,1.,1.]
        self.steps=0
        self.bound=runtime

    def state_dict(self):
        damage=np.zeros((6,6),np.float32)
        damage[0,3]=3.
        damage[2,3]=7.
        return {"cum_damage":damage}

    def step(self,action):
        self.steps+=1
        done=self.steps==self.bound
        info={"duration_s":.5,"executed_candidate":action,"override_reason":None}
        if done:
            info["episode"]={"win":True,"draw":False,"return":2.,"decisions":self.steps,
                "switches":1,"overrides":0,"recoveries":0}
        return self.cached,0.,done,info


def test_completed_trials_record_actual_controller_execution_and_diagnostic_fraction():
    result=run_control_trial(2,"fixed_center",{"seed":51001,"style":"capture","side":0},
                             env_factory=FixtureEnvironment)
    assert result["completed"] and not result["truncated"]
    assert result["actual_duration_s"]==1.
    assert result["diagnostics"]["candidate_counts"][1]==2
    assert result["diagnostics"]["executed_candidate_counts"][1]==2
    assert result["diagnostics"]["sampled_center_fraction"]==1.
    assert result["diagnostics"]["damage_share_of_team"]==pytest.approx(.7)
    assert result["SGD_steps"]==0 and result["exploratory_posthoc"]
    assert not result["independent_new_test"] and not result["real_nav2_in_loop"]


def test_unexpected_cutoff_raises_instead_of_reporting_draw():
    with pytest.raises(RuntimeError,match="no invented draw"):
        run_control_trial(3,"continue_only",{"seed":51001,"style":"pursuit","side":0},
                          env_factory=FixtureEnvironment,max_decisions=2)


def test_audit_refuses_existing_artifact_directory_before_creating_runtime(tmp_path):
    with pytest.raises(FileExistsError):
        audit(tmp_path)
