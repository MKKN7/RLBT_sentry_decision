Read-only documentation/report audit, 2026-11-17, temporal_baseline_agent.
Scope: docs/25_temporal_global_strategy_pilot.md, scripts/report_temporal_phase.py, registered config and delivered policy/trainer/evaluator. No implementation changed, no training/evaluation/ROS launched. Small read-only model parameter calculation only.

Decision: no material blocker to the registered pilot or honest reporting. Docs and generated report repeatedly distinguish simplified AlphaTaskEnv from real Point-LIO/Nav2/MPPI, official UL/UC, Combat XML and NUC. No superiority, full-season convergence or automatic permission claim. Stateful BT and GRU reset-every-step control are described with limits. Two-seed and unequal parameter-count limitations are explicit. Shared actor observation, candidates/mask and executor match delivered code; simulator truth used only in diagnostic accounting, not as additional actor inputs. The existing prototype HUD includes upstream enemy statistics, correctly called out by evaluator and future legal-observation requirement.

Verified exact facts:
- Instantiated TemporalActorCritic(memory=mlp/gru): 55,490 / 113,474 parameters, matching docs/report.
- Local fixture config episode_time=180.0, state has no ammo/coin fields; null/unimplemented economy claims appropriate.
- .997**180=.58227547199. Duration gamma/lambda, chronological full BPTT, separate actor/critic, trained candidate output and true terminal/truncation handling match current source.
- Registration is four runs x minimum4096 interactions, 2 model seeds, 8 test cases/policy, 8 policies=64 full episodes. Root runner labels/alias mapping and reporter unique-case validation now align. Registered noBC/no historical initialization preserved.
- Reporter now refuses existing output and validates raw training/evaluation completion/identity before making report directory; prior acceptance concerns in temporal_runner_report_review.md addressed.

Two optional wording precision fixes before publication:
1. docs/25 line35: “原文件hash在checkpoint及source快照中保存” overstates physical embedding. Current metadata stores upstream_manifest_sha256 plus upstream_commit; per-file ONNX/fixture hashes live in that verified local manifest, while source/ contains Python CONTRACT_SOURCES (not the manifest or ONNX). Prefer “上游文件由冻结manifest记录逐文件hash；checkpoint保存manifest hash，Python源码另存快照。” Runtime still verifies actual ONNX bytes, so this is provenance wording, not evidence loss.
2. scripts/report_temporal_phase.py line151 says “带Blackboard记忆的BT”. Delivered BTMemoryPolicy uses auditable Python instance state, no actual BehaviorTree.CPP Blackboard/XML runtime. Prefer “带内部状态的Python BT基线” to avoid implying XML integration. docs/25 already accurately calls it Python research baseline.

Minor pedagogical note: report line157 “强制单选只用于critic” means no direct loss at that timestep; with GRU, those observable inputs may still influence later actor decisions through hidden state. docs/25 line21 more accurately states excluded actor gradient but reward/time retained, though full BPTT can propagate through their state processing. Optional phrase “强制单选不计直接actor loss，仍保留时序与critic” is exact.
