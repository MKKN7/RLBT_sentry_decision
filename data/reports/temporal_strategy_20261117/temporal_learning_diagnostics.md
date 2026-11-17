# MLP learning diagnostics, 2026-11-17

Read-only scope: completed formal MLP runs seed271 and seed809 only. Current
GRU experiment left untouched. One CPU thread; no new environments, training,
SGD, test episodes or ROS were launched. All numbers below come from these
runs' initial.pt/final.pt, episodes.jsonl, updates.jsonl and saved rollout NPZ.
Machine-readable measurements: .work/temporal_learning_diagnostics.json.

## Measured execution and update signal

| Metric | MLP seed271 | MLP seed809 |
|---|---:|---:|
| Actual candidate decisions | 4116 | 4124 |
| Complete episodes / PPO updates | 56 / 14 | 51 / 13 |
| Truncated episodes | 0 | 0 |
| Actor decisions | 2920 (70.94%) | 2828 (68.57%) |
| Forced single-legal decisions | 1196 | 1296 |
| Minimum actor decisions in any update | 177 | 149 |
| Actor weight L2 change | 0.374989 | 0.363948 |
| Actor change / initial weight norm | 3.802% | 3.689% |
| Critic weight L2 change | 0.630194 | 0.764123 |
| Critic change / initial weight norm | 6.347% | 7.720% |
| Mean actor gradient norm before clipping | 0.03639 | 0.03282 |
| Mean critic gradient norm before clipping | 5.17830 | 3.29638 |
| Max first-epoch old/current approximate KL | 0 | 0 |
| Mean within-update approximate KL | 4.61e-7 | 2.51e-7 |
| PPO clipped fraction, every update | 0 | 0 |
| Mean recorded actor entropy | 2.772529 | 2.772493 |

All final parameters finite. This verifies executed actor learning, not just
critic movement or a disabled PPO path. Actor/critic clipping is separate, so
the larger critic norm does not by itself establish actor-gradient suppression.
The earlier single-valid-decision normalization edge case is absent here.

## Distribution movement on the SAME recorded training observations

These are training-only diagnostics; they are neither frozen performance
evaluation nor evidence of generalization. Compare each run's initial and
final actor on all that run's saved observations with >1 legal candidate.
Every such row has all16 legal; all other rows have exactly1 legal.

| Metric | seed271 | seed809 |
|---|---:|---:|
| Final mean entropy | 2.772463 | 2.772541 |
| Uniform16 entropy ln16 | 2.772589 | 2.772589 |
| Mean KL(final || uniform) | 0.0001260 | 0.00004779 |
| Mean KL(final || initial) | 0.00009286 | 0.00002979 |
| Mean total-variation(initial,final) | 0.005392 | 0.002865 |
| Final mean maximum action probability | 0.064184 | 0.063700 |
| Uniform16 action probability | 0.062500 | 0.062500 |
| Final mean top1-minus-top2 probability | 0.0003907 | 0.0003930 |
| Initial→final greedy label changed | 2920/2920 | 2286/2828 |

Seed271 final greedy is candidate8 on all2920 actor-valid recorded rows.
Its initial greedy is candidate2 on2776 rows and candidate10 on144 rows.
Seed809 final greedy candidate11 appears1355/2828 rows; other labels vary.
Sampled training actions remain spread across all16 choices: seed271 per-action
counts162–205, seed809157–192. Candidate IDs are the unchanged upstream task
schema (8=cover_east_upper,11=flank_southwest).

Inference: the actors have moved, but their probability distributions remain
very close to uniform. A large greedy-label change is possible when nearly
tied logits shift slightly, so that count alone does not show a learned useful
strategy. Zero clip fraction here is consistent with small distribution steps;
it is not evidence PPO clipping is broken. These small pilots do not establish
convergence or useful global decision-making.

## Actual temporal coverage

| Metric | seed271 | seed809 |
|---|---:|---:|
| Episode mean / median seconds | 36.566 / 34.25 | 40.202 / 35.10 |
| Episode min / max seconds | 25.0 / 65.4 | 22.7 / 68.0 |
| Total simulated seconds | 2047.7 | 2050.3 |
| Episodes reaching90s or179s | 0 / 0 | 0 / 0 |
| Train styles pretrained/capture/defense/resupply | 18/20/11/7 | 16/10/12/13 |
| Team0 / Team1 episodes | 26 / 30 | 29 / 22 |
| Proposed-task switches | 2523 | 2454 |
| BT overrides / recoveries | 1200 / 100 | 1300 / 110 |

Every saved sequence actually terminated in the frozen environment before68s;
there was no artificial400-decision truncation. Although the environment's time
limit is180s and the trainer keeps whole episodes, these completed MLP datasets
do not contain full180s trajectory experience. Early termination is not itself
a collector bug. Training wins32/56 and26/51 are descriptive on-policy outcomes
against changing training draws, not independent common-case performance.

The exact configured discount gamma=.997/sec has half-life230.70s and retains
0.5823 at180s. GAE additionally uses lambda=.97/sec: combined gamma*lambda
trace half-life20.71s, trace0.2622 at40s and0.002421 at180s. Inference: long-horizon
discounting and long-horizon practical advantage credit are different properties;
the stated gamma does not establish180s credit assignment by itself. This audit
does not propose changing any coefficient in the frozen pilot.

## Reward observations

The fixed `upstream` profile is unchanged for both runs. Tick team-average
reward is summed over each task interval, then .002 per switched task deducted.
No custom official UL/UC scoring/economy terms were added. Per completed episode:

| Mean absolute episode component | seed271 | seed809 |
|---|---:|---:|
| Terminal | 8.68063 | 8.25792 |
| Entire upstream dense reward | 0.64566 | 0.68871 |
| Switch cost | 0.09011 | 0.09624 |
| Own raw hit diagnostic | 0.03041 | 0.02881 |
| Low-HP-home shaping | 0.00534 | 0.00508 |

Terminal outcome dominates the absolute aggregate reward by this diagnostic.
Dense reward includes team combat/occupancy/contact signals; the learner
selects only one robot's task while the remaining fixed agents affect team
outcome. Inference: terminal-heavy team feedback plus many early endings may
make task credit noisy. This is a plausible mechanism, not a causal proof and
not an instruction to tune reward after viewing heldout results. The residual
contact_and_precision field remains a residual, not measured physical contact.

## Limits

This is the frozen simplified alphaRM3v3 prototype, not officialUL/UC maps,
rules/economy or a live Point-LIO/Nav2/MPPI episode. No final GRU rows or heldout
evaluation data used here. Frozen evaluation and GRU-reset ablation remain
necessary before any strategic or temporal performance conclusion.
