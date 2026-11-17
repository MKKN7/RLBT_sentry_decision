# Completed posthoc control audit, 2026-11-17

Implemented ONLY new script/tests and audit artifacts; frozen model,
environment, trainer, evaluator, reports and existing experiment unchanged.

- scripts/audit_temporal_policy_leverage.py
- tests/test_temporal_leverage.py
- runs/temporal_leverage_audit_20261117/manifest.json, episodes.jsonl,
  summary.json, status.json, plus exact copied source dependencies.
- .work/pytest_temporal_leverage.log, .work/temporal_leverage_audit.log,
  .work/temporal_leverage_source_verification.json.

Executed8tests: 8 passed in0.92s. Checks fixed mask-only selection, legal
fallbacks, invalid masks, actual execution+damage diagnostic fractions,
explicit cutoff error rather than invented draw, and existing-output refusal.

Actual audit command:
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/kellen/anaconda3/envs/robot/bin/python scripts/audit_temporal_policy_leverage.py --out runs/temporal_leverage_audit_20261117 --workers 2 --max-decisions 400
Launched detached subprocess PID2214087, completed normally and exited.
Actual16/16 complete episodes,0truncations, wall35.79s, no model weights loaded,
no optimizers created,0SGD. Each policy ran the exact8cases:
seeds51001/51002 × stylescapture/pursuit × sides0/1.

| Fixed controller | Wins | Mean return | Mean duration s | Alive fraction | Center fraction | Damage/team share |
|---|---:|---:|---:|---:|---:|---:|
| continue_only |8/8|11.07289|33.3500|1.0000|0.0000|0.09497|
| fixed_center |8/8|10.49829|30.6875|0.75596|0.66945|0.23179|

continue_only requested and executed action0 throughout, no switches,
overrides or recoveries. It stayed in home at sampled diagnostics, fullHP;
existing combat aiming/firing and fixed teammates continued normally.
This control deliberately does not disable the combat executor or teammates.
fixed_center requests1 whenever legal,0fallback; mean16.375BT overrides and
.875recoveries per episode, meanHP fraction.21806. All counts preserved raw.
Actual totals: continue536decisions/266.8s; fixedcenter494decisions/245.5s.
fixedcenter proposed action1=363,action0=131; final executed1=443,0=51.
Override reasons robot_dead124,blocked_abort_to_hold7. These follow unchanged
executor semantics, including active-intent continuation during action0/death.

Interpretation: even the mask-only continue controller wins all reused cases,
so the observed8/8 ceiling is not discriminating learned task-policy value in
this batch. Different controllers alter center presence, damage contribution,
survival and return, but win-rate ties cannot show PPO temporal/global gain.
This is an exploratory POSTHOC control on already-seen cases, not a new
independent test or policy-selection criterion. Do not combine into the
preregistered64episodes as though these controls were registered beforehand.
No causal proof that learned strategy is globally useless, no formalUL/UC or
realNav2 result. [1,1]seed-bootstrap here is degenerate at ceiling with only2
seeds, not strong generalization confidence.

All8dependency source hashes match both initial manifest and copied source
after execution. Existing learned checkpoints were never loaded or changed.
Controller function reads cached legalmask only; current simulator truth is
read solely for explicitly marked post-decision diagnostics.
