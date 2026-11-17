# Read-only temporal trainer audit, 2026-11-17

Scope: scripts/train_temporal_phase.py, tests/test_temporal_training.py,
current temporal policy interface, frozen AlphaTaskEnv transition contract.
No source edited, no ROS session or training launched by this audit.

## Decision

No blocker for the preregistered four ordinary 4096-decision pilot runs.
One P2 edge case was reproduced below. It is not present in the completed
smoke update (121 actor decisions), but future updates should log/check it.

## Finding

P2 edge case, scripts/train_temporal_phase.py:250-263: with exactly one valid
actor decision in the entire PPO batch, centering advantages sets that sole
advantage to zero. With entropy_coef=0 the actor receives no task-reward
gradient even for nonzero return; with entropy enabled it learns entropy only.
The same loss of relative-reward signal occurs for a batch of identical valid
advantages, as usual with centered advantage normalization. This is a learning
quality edge case, not numerical failure, hidden-state corruption or leakage.
Reproduction used one legal-choice terminal GRU decision, reward10, entropy0;
actual policy_loss0, actor_gradient_norm0, actor squared weight change0.
Suggested narrowly scoped future treatment: when valid_count<2 or advantage
std is below a documented floor, retain uncentered advantages (or explicitly
skip that actor update and report it), with a regression test. Do not silently
change the frozen experiment or reinterpret forced samples as actor choices.

## Verified by source and current tests

- Collection carries separate actor/critic hidden states chronologically;
  first-observation reset only, no optimizer step during episode collection.
- PPO re-evaluates each complete sequence under current weights from zero
  hidden on each epoch; no shuffled individual timesteps, stale hidden reuse,
  or detach inside sequences. Full BPTT stays connected across observations.
- First-epoch old/current logp equivalence is tested for MLP and GRU.
- Duration GAE uses gamma**duration and lambda**duration. True termination
  zeroes bootstrap, explicit decision-limit truncation keeps a next-observation
  recurrent value using the last collected hidden, then cuts trace.
- Only sampled actions and their recorded legal masks enter policy ratios.
  Forced single-legal decisions are excluded from actor loss but retained in
  value targets, reward timing and earlier choices' return. Existing Adam
  momentum cannot change actor in an all-forced update (grads set None).
- Whole-episode losses normalize once by total valid actor/all critic rows;
  optimizer steps once after all sequence backward passes in an epoch.
- Actor/critic parameter sets and gradient clipping are separate. The value
  loss has no actor parameter path. No environment truth enters model inputs.
- Non-finite observations/losses/gradient norms and early interior terminal
  flags are rejected. Heldout pursuit is rejected before collection steps.
- Actual completed smoke: 142 decisions,121 actor,21 forced,2 true completed
  episodes,0 truncations,first_epoch_KL0,finite parameters and changed weights.

Executed: OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
/home/kellen/anaconda3/envs/robot/bin/python -m pytest -q tests/test_temporal_training.py
Result: 15 passed in 4.10s. Additional one-decision reproduction above executed.

Not re-executed here: full regression, four final pilot training runs, evaluator,
official UL/UC simulation, real Nav2 or hardware. Unit/mock success is not a
claim of PPO strategic superiority or temporal/global performance gains.
