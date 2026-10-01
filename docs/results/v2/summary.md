# FastLane v2 results (MCP-Bench, real tools)

10 test tasks (0 with live A/B/C traces). τ = 0.8. Spend: baseline $0.048, live $0.085, replay $0.002.

## Primary endpoints (live)

Geometric-mean ratio of agent time per episode over tasks (below 1 is faster), 95% bootstrap CI over tasks, two-sided sign-flip p.

| Comparison | Ratio | 95% CI | p |
|---|---|---|---|
| B / A (sPTC) | 0.767 | [0.619, 0.967] | 0.038 |

## Live arms

| Arm | Episodes | Mean agent time s | p50 turn s | p90 turn s | Hits / READ calls | Wasted tool-s per s saved vs A | Call amplification | Distinct spec. entities |
|---|---|---|---|---|---|---|---|---|
| A (baseline) | 18 | 20.9 | 16.11 | 38.51 | 0/379 | n/a | 1.00 | 0 |
| B (sPTC) | 18 | 16.1 | 14.07 | 30.55 | 298/440 | 0.00 | 1.00 | 0 |

Amplification = (speculative launches + real executions) / calls; wasted tool-s are unused launches, t_end - t_launch.

### Speculative failures by kind

| Arm | Launches | Abandoned (unused) | Failed | By kind | Failure rate |
|---|---|---|---|---|---|
| B | 298 | 0 | 3 | tool_error 3 | 0.010 |

## Replay primary

Does prediction quality matter beyond starting calls early? Ratios below 1 favour the predictor.

| Comparison | Ratio | 95% CI | p |
|---|---|---|---|
| pred_frozen / frequency | 1.000 | [1.000, 1.000] | 1.000 |
| pred_frozen / random_tool | 1.000 | [1.000, 1.000] | 1.000 |

## Ladder (replay)

| Arm | Agent time vs A (geo-mean) | Hits / READ | Launches | Wasted tool-s per s saved | Amplification | Distinct spec. entities |
|---|---|---|---|---|---|---|
| A | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |
| random_tool | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |
| frequency | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |
| wrong_arg | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |
| pred_frozen | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |
| exhaustive_2 | 0.969 | 3/123 | 160 | 13.49 | 2.27 | 148 |
| exhaustive_4 | 0.944 | 8/123 | 295 | 14.30 | 3.31 | 261 |
| exhaustive_8 | 0.920 | 10/123 | 486 | 18.87 | 4.84 | 420 |
| oracle | 0.764 | 123/123 | 123 | 0.00 | 1.00 | 0 |
| pred_online | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |
| guard | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |
| self | 0.947 | 21/123 | 27 | 0.27 | 1.05 | 4 |
| pred_no_slowdown | 1.000 | 0/123 | 0 | n/a | 1.00 | 0 |

## Seen vs unseen server

| Test tasks | n | pred_frozen / A |
|---|---|---|
| seen (all servers in train) | 10 | 1.000 |
| unseen (a server not in train) | 0 | n/a |

## Hit rate by argument source

| Argument source | READ calls | Hits | Hit rate |
|---|---|---|---|
| user | 27 | 0 | 0.00 |
| result | 7 | 0 | 0.00 |
| model | 89 | 0 | 0.00 |

## Learning curves (history)

| History | τ chosen | Agent time vs A | Hits / READ |
|---|---|---|---|
| 0 train episodes (frozen) | 0.8 | 1.000 | 0/123 |
| 5 train episodes (frozen) | 0.8 | 1.000 | 0/123 |
| 10 train episodes (frozen) | 0.8 | 1.000 | 0/123 |
| all train, frozen | 0.8 | 1.000 | 0/123 |
| all train + online | 0.8 | 1.000 | 0/123 |

## τ curve

Measured on test, reported only, never used to choose τ.

| τ | Agent time vs A | Hit rate | Wasted tool-s per s saved |
|---|---|---|---|
| 0.05 | 0.986 | 0.07 | 4.22 |
| 0.1 | 0.986 | 0.07 | 4.21 |
| 0.2 | 1.000 | 0.01 | 206.36 |
| 0.3 | 1.000 | 0.00 | n/a |
| 0.4 | 1.000 | 0.00 | n/a |
| 0.6 | 1.000 | 0.00 | n/a |
| 0.8 | 1.000 | 0.00 | n/a |

## Delay ratio

| Scale | Tool / LLM latency | pred_frozen / A | oracle / A |
|---|---|---|---|
| 0.25 | 0.03 | 1.000 | 0.920 |
| 0.5 | 0.06 | 1.000 | 0.857 |
| 1 | 0.12 | 1.000 | 0.764 |
| 2 | 0.25 | 1.000 | 0.654 |
| 4 | 0.49 | 1.000 | 0.585 |

## Replay check (live P vs replay)

_Skipped: inputs missing._

## Per-class results

| Class | Tasks | Replay pred_frozen / A | Live C / B |
|---|---|---|---|
| D0 | 3 | 1.000 | n/a |
| D1 | 7 | 1.000 | n/a |
| F1 | 2 | 1.000 | n/a |
| F2 | 8 | 1.000 | n/a |
| X | 1 | 1.000 | n/a |

## Falsification checklist

Each line is a condition that would count against a positive C result; yes means it holds.

- frequency within 5% of the predictor (or faster): **yes** (frequency / pred_frozen = 1.000)
- wrong_arg within 5% of the predictor (or faster): **yes** (wrong_arg / pred_frozen = 1.000)
- live C fails to beat B while replay says it should: **n/a** (live C/B = n/a, replay pred_frozen/A = 1.000)
- live C vs replay disagreement (info): live C/A = n/a, replay pred_frozen/A = 1.000
- gain concentrated in 2-3 tasks, replay pred_frozen vs A (top-3 share above 50%): **n/a** (top-3 share of total saved time = n/a over 10 tasks)
- gain concentrated in 2-3 tasks, live C vs B (top-3 share above 50%): **n/a** (top-3 share of total saved time = n/a over 0 tasks)
- C trajectories differ from A/B on identical tool outputs (info, LLM sampling can also differ): real-call sequences equal to A on B 4/18
- gain tracks the number of speculative calls rather than hit quality (across ladder arms): **no** (corr(saved, launches) = 0.47, corr(saved, hits) = 0.94)
- gain appears only on seen servers or transitions: **n/a** (pred_frozen/A seen servers 1.000, unseen n/a; hits on seen calls 0/95, new calls 0/28)

## Structure

### Structure of baseline traces

| task | split | servers | calls | fan-out | chain depth | classes | ceiling |
|---|---|---|---|---|---|---|---|
| syn_okx_01 | train | okx_exchange | 1 | 1 | 1 | F0 D0 | 0.000 |
| syn_nixos_03 | train | nixos | 34 | 34 | 1 | F2 D0 | 0.618 |
| game_trends_000 | train | game_trends | 9 | 7 | 1 | F2 D0 | 0.415 |
| syn_nixos_05 | train | nixos | 93 | 11 | 2 | F2 D1 | 0.793 |
| syn_okx_03 | train | okx_exchange | 10 | 7 | 2 | F2 D1 | 0.166 |
| okx_exchange_001 | train | okx_exchange | 2 | 2 | 1 | F1 D0 | 0.007 |
| syn_nixos_04 | train | nixos | 26 | 5 | 2 | F2 D1 | 0.368 |
| syn_x_01 | train | game_trends,nixos | 80 | 35 | 3 | F2 D2 X | 0.585 |
| syn_nixos_01 | test | nixos | 4 | 2 | 2 | F1 D1 | 0.093 |
| syn_okx_02 | test | okx_exchange | 15 | 5 | 2 | F2 D1 | 0.121 |
| nixos_000 | test | nixos | 12 | 9 | 2 | F2 D1 | 0.360 |
| syn_game_01 | test | game_trends | 7 | 5 | 1 | F2 D0 | 0.398 |
| syn_nixos_06 | test | nixos | 5 | 3 | 2 | F2 D1 | 0.264 |
| syn_game_02 | test | game_trends | 4 | 2 | 1 | F1 D0 | 0.275 |
| syn_nixos_07 | test | nixos | 9 | 5 | 2 | F2 D1 | 0.274 |
| okx_exchange_000 | test | okx_exchange | 12 | 12 | 1 | F2 D0 | 0.097 |
| syn_nixos_02 | test | nixos | 14 | 7 | 2 | F2 D1 | 0.324 |
| syn_x_02 | test | nixos,okx_exchange | 42 | 24 | 2 | F2 D1 X | 0.430 |

#### train (n=8)

- Ceiling: median 0.392, mean 0.369
- Calls by argument source: user 16%, result 47%, model 36%
- Episodes with fan-out >= 2: 88%
- Episodes with a result-to-argument hop: 50%
- Chain depth: median 1.5, p90 2.3
- Classes: D0 4, D1 3, D2 1, F0 1, F1 1, F2 6, X 1


#### test (n=10)

- Ceiling: median 0.275, mean 0.264
- Calls by argument source: user 23%, result 6%, model 72%
- Episodes with fan-out >= 2: 100%
- Episodes with a result-to-argument hop: 70%
- Chain depth: median 2.0, p90 2.0
- Classes: D0 3, D1 7, F1 2, F2 8, X 1


## Calibration

Slowdown factor of a call when k identical calls run at once.

| Tool | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| game_trends__get_all_trending_games | 1.00 | 0.90 | 1.10 | 1.10 |
| game_trends__get_epic_free_games | 1.00 | 0.06 | 0.05 | 0.07 |
| game_trends__get_epic_trending_games | 1.00 | 0.33 | 0.33 | 0.32 |
| game_trends__get_steam_most_played | 1.00 | 1.24 | 1.12 | 1.58 |
| game_trends__get_steam_top_sellers | 1.00 | 2.95 | 2.81 | 3.00 |
| game_trends__get_steam_trending_games | 1.00 | 1.11 | 1.12 | 1.07 |
| nixos__darwin_info | 1.00 | 1.05 | 1.37 | 2.56 |
| nixos__darwin_options_by_prefix | 1.00 | 1.10 | 1.26 | 2.39 |
| nixos__darwin_search | 1.00 | 1.08 | 1.20 | 2.38 |
| nixos__home_manager_info | 1.00 | 1.10 | 1.24 | 2.56 |
| nixos__home_manager_search | 1.00 | 1.27 | 1.43 | 2.86 |
| nixos__nixhub_find_version | 1.00 | 0.59 | 0.72 | 1.59 |
| nixos__nixhub_package_versions | 1.00 | 0.90 | 0.96 | 1.84 |
| nixos__nixos_channels | 1.00 | 1.17 | 0.77 | 1.04 |
| nixos__nixos_info | 1.00 | 0.98 | 0.97 | 1.98 |
| nixos__nixos_search | 1.00 | 1.06 | 1.00 | 1.97 |
| nixos__nixos_stats | 1.00 | 0.95 | 0.92 | 1.49 |
| okx_exchange__get_candlesticks | 1.00 | 0.86 | 1.00 | 1.01 |
| okx_exchange__get_price | 1.00 | 0.86 | 0.88 | 0.90 |
