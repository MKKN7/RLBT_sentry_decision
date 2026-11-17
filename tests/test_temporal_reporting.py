"""Reject incomplete or mismatched experiments before publishing comparisons."""
import copy

import pytest

from scripts.report_temporal_phase import validate_registered_inputs


def registered_artifacts():
    registration = {'memory_variants':['mlp','gru'], 'training_seeds':[271,809],
        'minimum_decisions_per_run':4096, 'gamma_per_second':.997, 'lambda_per_second':.97,
        'episodes_per_update':4, 'epochs':2, 'learning_rate':.0003, 'entropy_coefficient':.01,
        'test_seeds':[51001,51002], 'test_styles':['capture','pursuit'], 'test_sides':[0,1],
        'test_policies':['reactive_BT','memory_BT','mlp_s271','mlp_s809','gru_s271','gru_s809',
                         'gru_reset_each_step_s271','gru_reset_each_step_s809']}
    training, checkpoints = [], []
    for memory in ('mlp','gru'):
        for seed in (271,809):
            label = f'{memory}_s{seed}'
            config = {'memory':memory,'seed':seed,'gamma_per_second':.997,'lambda_per_second':.97,
                'episodes_per_update':4,'epochs':2,'learning_rate':.0003,'entropy_coef':.01,
                'target_decisions':4096,'BC_updates':0,'randomization':True,'reward_profile':'upstream',
                'real_nav2_in_loop':False,'official_UL_UC_training':False}
            training.append({'run':label,'config':config,
                'summary':{'status':'complete','decisions':4100,'completed_training_episodes':2,'truncated_sequences':0},
                'episodes':[{'decisions':2050,'terminated':True,'truncated':False} for _ in range(2)],
                'updates':[{'batch_decisions':4100}], 'source_sha256':{'final.pt':label+'_digest'}})
            checkpoints.append({'label':label,'sha256':label+'_digest','weights_unchanged':True})
    labels = ['bt_reactive','bt_memory','mlp_s271','mlp_s809','gru_s271','gru_s809','gru_s271_reset','gru_s809_reset']
    episodes = [{'policy':label,'seed':seed,'style':style,'side':side,'completed':True,'truncated':False}
        for label in labels for seed in (51001,51002) for style in ('capture','pursuit') for side in (0,1)]
    evaluation = {'status':'complete','expected_episodes':64,'actual_episodes':64,'SGD_steps':0,
        'checkpoints':checkpoints,'real_nav2_in_loop':False,'official_UL_UC_training':False}
    return registration, training, evaluation, episodes


def test_exact_registered_experiment_maps_logical_policy_aliases():
    aliases = validate_registered_inputs(*registered_artifacts())
    assert aliases['reactive_BT'] == 'bt_reactive'
    assert aliases['gru_reset_each_step_s271'] == 'gru_s271_reset'


@pytest.mark.parametrize('corruption', ['missing_case','duplicate_case','wrong_seed','wrong_checkpoint',
    'incomplete_run','truncated_training','wrong_raw_count','missing_update','wrong_discount',
    'evaluated_updates','changed_weights','truncated_evaluation','scope_flags','missing_train_seed'])
def test_report_rejects_corrupt_or_unregistered_evidence(corruption):
    registration, training, evaluation, episodes = registered_artifacts()
    if corruption == 'missing_case': episodes.pop()
    elif corruption == 'duplicate_case': episodes[-1] = copy.deepcopy(episodes[0])
    elif corruption == 'wrong_seed': episodes[-1]['seed'] = 9
    elif corruption == 'wrong_checkpoint': evaluation['checkpoints'][0]['sha256'] = 'other_digest'
    elif corruption == 'incomplete_run': training[0]['summary']['status'] = 'running'
    elif corruption == 'truncated_training': training[0]['episodes'][0].update(terminated=False,truncated=True)
    elif corruption == 'wrong_raw_count': training[0]['episodes'][0]['decisions'] -= 1
    elif corruption == 'missing_update': training[0]['updates'] = []
    elif corruption == 'wrong_discount': training[0]['config']['gamma_per_second'] = .9
    elif corruption == 'evaluated_updates': evaluation['SGD_steps'] = 1
    elif corruption == 'changed_weights': evaluation['checkpoints'][0]['weights_unchanged'] = False
    elif corruption == 'truncated_evaluation': episodes[0].update(completed=False,truncated=True)
    elif corruption == 'scope_flags': evaluation['real_nav2_in_loop'] = True
    elif corruption == 'missing_train_seed': training[-1]['config']['seed'] = 271
    with pytest.raises(ValueError):
        validate_registered_inputs(registration,training,evaluation,episodes)
