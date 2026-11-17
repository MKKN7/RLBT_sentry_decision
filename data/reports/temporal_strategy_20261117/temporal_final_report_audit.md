# Final recorded pilot audit, 2026-11-17

Read-only temporal_baseline_agent audit of data/reports/temporal_strategy_20261117/{comparison.json,report.md}, raw runs/temporal_frozen_evaluation_20261117/{episodes.jsonl,summary.json}, four training artifacts and checkpoint bytes. No training, environment, evaluation or ROS process launched; source untouched.

## Acceptance

No artifact-integrity or report-number blocker found. 64 raw unique policy/seed/style/side identities exactly equal registered8 policies x seeds51001/51002 x styles capture/pursuit x sides0/1. 64 true complete terminal episodes, zero unexpected truncations, SGD_steps0. All four serialized checkpoint hashes and reconstructed model tensor hashes independently match evaluator before/after hashes. Report tables agree with recomputed raw metrics. Source official/live flags all false.

| Run | decisions | true full episodes | PPO updates | actor / forced | zero single-actor updates |
|---|---:|---:|---:|---:|---:|
| mlp_s271 | 4116 | 56 | 14 | 2920 /1196 | yes |
| mlp_s809 | 4124 | 51 | 13 | 2828 /1296 | yes |
| gru_s271 | 4133 | 55 | 14 | 2859 /1274 | yes |
| gru_s809 | 4177 | 50 | 13 | 2951 /1226 | yes |

Per-run decisions independently sum exactly from raw episodes and update batches; terminal counts equal all raw training rows, zero truncations. Report input hashes for config, summary, episodes, updates, initial/final.pt all match current bytes. Previously known single-actor-advantage issue never triggered.

## Measured evaluation

All eight policies won8/8, zero draws. This is a saturated, low-discrimination test batch, not evidence of PPO improvement. Mean return / sampled center fraction / own damage:
- reactiveBT: 11.37465 / .51045 /391.80
- memoryBT: 11.05982 / .48470 /351.38
- MLP271: 10.65589 /0 /236.94
- MLP809: 11.01210 /0 /303.55
- GRU271: 11.07289 /0 /133.17
- GRU809: 10.65589 /0 /236.94
- Each GRU reset ablation exactly matches carried GRU outcomes and returns on every one of8 cases.

Across64 episodes actual durations21.7–46.1sec, not180sec trajectories. Four training groups max22.5–83.2sec depending run. This dataset cannot demonstrate long full-season timing/global planning.

Behavior is particularly important: GRU271 proposed CONTINUE/HOLD (candidate0) on all536 evaluation decisions, zero task switches, center fraction0. MLP271 and GRU809 proposed cover candidate8 whenever alive (444 times, remaining94 forced0); MLP809 proposed flank candidate11 whenever alive (418 times, remaining70 forced0). Thus100% team wins coexist with weak/constant learner task choice. Fixed teammates and weak selected style cases can dominate outcome. This is a useful measured ceiling diagnosis, not a claim that all BTs are optimal or PPO incapable.

## Memory ablation validity

Each carried GRU resets once per episode; corresponding reset variant hidden_reset_count equals every episode's decision count. Same first-case NPZ arrays state/candidates/mask/action/executed/reward/duration/terminated match exactly for carried/reset for both GRUs. Reset ablation was actually applied, yet frozen greedy behavior/result insensitive on these cases. That supports only no measured useful memory dependence in the recorded cases; hidden/probabilities may still differ and unobserved states remain outside this result.

## Interpretation / publication

Report correctly states simplified alphaRM proxy, no realNav2, no officialULUC, independent actor/critic, unequal MLP/GRU capacity and hidden-reset distribution shift. No causal/PPO superiority or convergence claim. If final user summary is brief, explicitly include saturated8/8 for BOTHBTs and PPO, plus GRUreset no difference; reporting8/8 PPO alone would be misleading. Consider mentioning GRU271 held still and short durations as next design signal. Next experiment should increase diagnostic challenge and strategic contribution without weakening fair statefulBT or safety, and require more seeds/longer informative trajectories before any global superiority claim.
