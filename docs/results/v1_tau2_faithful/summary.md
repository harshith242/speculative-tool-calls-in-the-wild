# FastLane results

12 tau2 retail tasks. τ = 0.05. Live spend $0.181, replay spend $0.018. Cost per correct answer $0.005 (live spend / correct episodes, all arms; speculation does not change LLM spend). 0 turns needed an API retry.

## Live arms (reality check)

| Arm | Turns | p50 s | p95 s | Mean agent time / episode s | Hits / READ calls | Wasted / real tool-s | Abandoned / leaked IDs | Success |
|---|---|---|---|---|---|---|---|---|
| A (baseline) | 63 | 4.36 | 22.20 | 33.3 | 0/87 | 0.00 | 0 / 0 | 0.92 |
| B (sPTC) | 61 | 4.07 | 14.49 | 25.5 | 77/77 | 0.04 | 4 / 0 | 1.00 |
| C (sPTC + predictor) | 58 | 3.63 | 14.54 | 26.0 | 92/92 | 0.83 | 68 / 11 | 1.00 |

## Primary endpoint: agent time per episode, live (geometric-mean ratio, 95% CI over tasks, sign-flip p)

| Comparison | Ratio | 95% CI | p |
|---|---|---|---|
| B / A | 0.806 | [0.729, 0.895] | 0.006 |
| C / B | 0.984 | [0.908, 1.075] | 0.734 |
| C / A | 0.793 | [0.708, 0.890] | 0.005 |

## Live paired differences, secondary (task-clustered 95% CI, seconds; p95 rests on few tasks)

| Comparison | p50 diff | p95 diff |
|---|---|---|
| C − B | -0.43 [-1.30, +0.34] | +0.04 [-4.67, +6.82] |
| B − A | -0.29 [-0.77, +1.04] | -7.70 [-16.55, -1.09] |
| C − A | -0.73 [-1.55, +0.50] | -7.66 [-16.09, -0.77] |

## Replay arms: prediction strategies on identical baseline conversations (order 0, no sPTC)

| Arm | Turns | p50 s | p95 s | Mean agent time / episode s | Hits / READ calls | Wasted / real tool-s | Abandoned / leaked IDs |
|---|---|---|---|---|---|---|---|
| A | 63 | 4.36 | 22.18 | 33.2 | 0/87 | 0.00 | 0 / 0 |
| pred_alone | 63 | 3.78 | 17.33 | 26.7 | 47/87 | 0.69 | 54 / 6 |
| random_k | 63 | 4.13 | 20.75 | 30.0 | 26/87 | 2.03 | 173 / 124 |
| oracle | 63 | 3.76 | 10.37 | 22.1 | 87/87 | 0.06 | 5 / 0 |
| guard | 63 | 3.78 | 17.33 | 27.1 | 44/87 | 0.51 | 41 / 3 |
| self | 63 | 4.07 | 15.30 | 28.0 | 57/87 | 0.16 | 16 / 3 |

## Replay paired differences (task-clustered 95% CI, seconds)

| Comparison | p50 diff | p95 diff |
|---|---|---|
| pred_alone − A | -0.57 [-1.60, +0.00] | -4.85 [-10.53, -0.76] |
| random_k − A | -0.22 [-0.72, +0.00] | -1.43 [-6.85, -1.34] |
| pred_alone − random_k | -0.35 [-1.47, +0.00] | -3.42 [-5.25, +0.68] |
| oracle − A | -0.59 [-1.09, +0.00] | -11.81 [-17.96, -7.66] |
| guard − pred_alone | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +2.88] |
| self − A | -0.28 [-0.59, +0.00] | -6.88 [-9.23, -2.43] |
| pred_alone − self | -0.29 [-1.29, +0.16] | +2.03 [-4.49, +2.41] |

## Seen-before vs new READ calls (replay, order 0)

| Arm | Hit rate, seen | Hit rate, new |
|---|---|---|
| pred_alone | 46/82 | 1/5 |
| guard | 43/82 | 1/5 |

## Order spread (replay): predictor alone − A, p50 and p95 per task order

- order 0: p50 -0.57 s, p95 -4.85 s
- order 1: p50 -0.57 s, p95 -4.85 s
- order 2: p50 -0.57 s, p95 -4.85 s

## Hit@k before each code step (first call of the script, exact match)

| Source | Steps | Hit@1 | Hit@3 |
|---|---|---|---|
| pred | 75 | 0.36 | 0.41 |
| self | 75 | 0.87 | 0.87 |

## τ sweep (predictor alone, replay)

| τ | Hit rate | Wasted / real tool-s | p50 s | p95 s |
|---|---|---|---|---|
| 0.05 | 0.54 | 0.69 | 3.78 | 17.33 |
| 0.1 | 0.54 | 0.64 | 3.78 | 17.33 |
| 0.15 | 0.54 | 0.64 | 3.78 | 17.33 |
| 0.2 | 0.52 | 0.56 | 3.78 | 17.33 |
| 0.3 | 0.44 | 0.49 | 3.78 | 17.33 |
| 0.4 | 0.44 | 0.42 | 3.78 | 17.33 |
| 0.5 | 0.43 | 0.42 | 3.78 | 17.33 |
| 0.6 | 0.39 | 0.28 | 4.13 | 17.33 |
| 0.8 | 0.28 | 0.13 | 4.13 | 18.19 |

## Simulator check: per-task agent time, replay vs live

- A: median absolute error 0.1%
