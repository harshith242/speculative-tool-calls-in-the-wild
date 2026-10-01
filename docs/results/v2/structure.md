# Structure of baseline traces

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

## train (n=8)

- Ceiling: median 0.392, mean 0.369
- Calls by argument source: user 16%, result 47%, model 36%
- Episodes with fan-out >= 2: 88%
- Episodes with a result-to-argument hop: 50%
- Chain depth: median 1.5, p90 2.3
- Classes: D0 4, D1 3, D2 1, F0 1, F1 1, F2 6, X 1


## test (n=10)

- Ceiling: median 0.275, mean 0.264
- Calls by argument source: user 23%, result 6%, model 72%
- Episodes with fan-out >= 2: 100%
- Episodes with a result-to-argument hop: 70%
- Chain depth: median 2.0, p90 2.0
- Classes: D0 3, D1 7, F1 2, F2 8, X 1

