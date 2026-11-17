from argparse import Namespace

import numpy as np
import pytest
import torch

from rl_bt.temporal_policy import TemporalActorCritic
from scripts import evaluate_strategy_v2 as evaluator


def observation():
    state=np.zeros(148,np.float32)
    state[[7,11,125]]=1
    candidates=np.zeros((16,16),np.float32)
    candidates[:,0]=np.arange(16)/16
    return {'state':state,'candidates':candidates,'mask':np.ones(16,bool)}


class FakeEnv:
    calls=[]
    def __init__(self,runtime,seed,**kwargs):
        self.calls.append((seed,kwargs.copy()))
        self.cached=observation()
        self.learner=2+3*kwargs['side']
        self.dropout=np.zeros(3,bool)
        self.obs=np.zeros((6,140),np.float32)
        self.obs[:,[7,11]]=1
        self.obs[self.learner,21]=1
        self.steps=0

    def state_dict(self):
        return {'cum_damage':np.ones((6,6),np.float32),'score':np.asarray([2.,1.])}

    def step(self,action):
        self.steps+=1
        return observation(),1.,self.steps==2,{'duration_s':.5,'executed_candidate':action,
            'override_reason':None,'episode':{'return':2.,'decisions':2,'switches':0,'overrides':0,
                'recoveries':0,'win':True,'draw':False,'winner':1,'outcome':1,'score':[2.,1.]}}


def arguments(tmp_path,checkpoint):
    return Namespace(seeds=[73001],styles=['pretrained'],sides=[0,1],
        checkpoint=[f'gru_s1701={checkpoint}'],support_mode='anchors',out=tmp_path/'evaluation',
        split='test',workers=1,max_decisions=4)


def test_v2_evaluator_routes_every_policy_to_new_factory_and_freezes_weights(tmp_path,monkeypatch):
    checkpoint=tmp_path/'final.pt'
    checkpoint.write_bytes(b'local-test-identity')
    model=TemporalActorCritic(memory='gru')
    payload={'metadata':{'support_mode':'anchors','memory':'gru','seed':1701},'decisions':8192}
    monkeypatch.setattr(evaluator,'load_strategy_checkpoint',lambda path:(model,payload))
    monkeypatch.setattr(evaluator,'RoleBalancedTaskEnv',FakeEnv)
    monkeypatch.setattr(evaluator,'AlphaRuntime',lambda:Namespace(source_commit='test'))
    FakeEnv.calls=[]
    result=evaluator.evaluate(arguments(tmp_path,checkpoint))
    assert result['actual_episodes']==12 # 2 cases x 4 controls/BT + GRU + reset.
    assert len(FakeEnv.calls)==12
    assert all(call[1]['support_mode']=='anchors' for call in FakeEnv.calls)
    assert all(row['completed'] and not row['truncated'] for row in result['episodes'])
    assert result['SGD_steps']==0 and result['checkpoints'][0]['weights_unchanged']
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not any(row.get('exploratory_posthoc',False) for row in result['episodes'])
    resets=[row['hidden_reset_count'] for row in result['episodes'] if row['policy']=='gru_s1701_reset']
    assert resets==[2,2]
    assert (tmp_path/'evaluation/gru_s1701_seed73001_pretrained_side0.npz').is_file()


def test_v2_evaluator_rejects_wrong_scene_mode_before_output(tmp_path,monkeypatch):
    checkpoint=tmp_path/'final.pt'
    checkpoint.write_bytes(b'test')
    monkeypatch.setattr(evaluator,'load_strategy_checkpoint',lambda path:(TemporalActorCritic(),
        {'metadata':{'support_mode':'other'}}))
    with pytest.raises(ValueError,match='support mode'):
        evaluator.evaluate(arguments(tmp_path,checkpoint))
    assert not (tmp_path/'evaluation').exists()
