# Independent read-only live actor validity review

Reviewed current source, source snapshots, training runtime evidence and actual transition/update JSONL for `runs/uc_live_actor_validity_train_20261117`. No source files modified, no ROS/simulator/test/training process launched by this reviewer. `AGENTS.md` search found no repository-local file. Memory lookup had no project hits.

## Verdict

No blocking learning-mechanics defect found in this revision. The live helper excludes only unapplied proposals and applied observations with one legal action from actor normalization, PPO surrogate, entropy and pre/post policy diagnostics. All actual transitions remain in duration GAE and critic regression. Actor and critic parameter sets are independently checked. No actor optimizer step occurs when both on-policy choices and informative anchors are absent, avoiding Adam momentum drift. Anchor-only updates are separately identified and require train-demonstration role plus a frozen teacher. The live trainer propagates explicit Boolean `action_applied=False` from terminal-before-admission feedback; ordinary BT-vetoed proposals retain actor credit as attempted intents.

## Verified actual training counts

- Complete, 8 episodes; 108 real RPC transitions; 14 rollout updates; 5 local task successes.
- 26 unique actor-eligible transitions, 82 applied single-choice HOLD transitions, 0 unapplied terminal-before-admission transitions. All 108 used as critic samples.
- 28 actor optimizer steps across 2 PPO epochs; 12 included on-policy signal, 16 anchor-only. All 8 rollout updates without eligible actor rows report exactly zero on-policy gradient norm.
- 310 logged MPPI command samples, 305 nonzero; all 8 copied sessions pass existing actual runtime verification.
- All current implementation files hash-match the frozen training source snapshot.
- Actual simulation elapsed time sums to 52.085 seconds. No exact physical resume.
- Per-episode eligible counts: `[4, 0, 4, 2, 4, 6, 0, 6]`. The two 40-step timeout episodes had only HOLD legal from the first observation; they must not be described as policy choosing inactivity.
- Successful episodes selected task candidate 0 and then CONTINUE; the remaining early failure selected an intermediate candidate, received one BT veto, later selected task0, and was safety-canceled. No success claim inferred merely from proximity.

## Attribution limitations for report

The 5/8 training success rate is not a controlled comparison with the prior 1/8 run: startup support and declared sensor seeds changed. The registered final and BC-initial greedy evaluations use the same sensor-seed order, but scheduling/physics/sensor queues are not identically paired. Results remain noisy local navigation engineering tasks and must not be called match win rates or proof of PPO advantage.

BC has only 6 informative decisions (1 navigate, 5 continue) plus 122 forced HOLD labels and no recovery labels. Its 100% training accuracy does not establish broad policy competence. The total actor delta 5.379 includes BC; PPO-only actor delta is 0.0559705. Fixed initial/final checkpoint evaluations, every startup attempt, and all unsupported sessions must stay in the denominator. No checkpoint selection from these evaluations.

Observed full regression log `.work/pytest_live_validity_full.log` reports 278 passed in 44.82 seconds. This reviewer did not rerun the suite.

Review completed while independent frozen BC evaluation was still running; this review does not pre-judge its final metrics.
