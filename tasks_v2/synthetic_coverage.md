# Task set coverage (18 tasks: 4 MCP-Bench originals + 14 synthetic)

Layout: `mcpbench_originals.json` = okx_exchange_000, okx_exchange_001, game_trends_000 (unchanged) + nixos_000 (revised, `fl_revised` set). `synthetic.json` = 14 synthetic tasks (nixos_001 and game_trends_001 replaced by syn_nixos_06 / syn_nixos_07). Synthetic grounding used 80 real tool calls, this revision 14 more. Live data (prices, giveaways, top sellers) will drift.

| task_id | structure | arg sources | length (realised calls) | intent | servers |
|---|---|---|---|---|---|
| game_trends_000 (orig) | fanout | none | medium (6) | plan | Game Trends |
| okx_exchange_000 (orig) | mixed | user, model | medium (~6) | monitor | OKX |
| okx_exchange_001 (orig) | mixed | user, model | short (~4) | monitor | OKX |
| nixos_000 (orig, revised) | mixed | user, result | medium (7) | plan | NixOS |
| syn_nixos_01 | single | user, model | short (1) | lookup | NixOS |
| syn_nixos_02 | mixed | user, result, model | medium (4) | troubleshoot | NixOS |
| syn_nixos_03 | fanout | user | medium (6) | compare | NixOS |
| syn_nixos_04 | mixed | user, result, model | long (10) | plan | NixOS |
| syn_nixos_05 | chain1 | user, result, model | short (2) | lookup | NixOS |
| syn_nixos_06 | chain2plus | user, result | medium (5) | lookup | NixOS |
| syn_nixos_07 | conditional | user, result, model | short (3-4) | troubleshoot | NixOS |
| syn_okx_01 | single | user, model | short (1) | lookup | OKX |
| syn_okx_02 | fanout | user, model | long (10) | compare | OKX |
| syn_okx_03 | conditional | user, result, model | short (3) | troubleshoot | OKX |
| syn_game_01 (revised) | fanout | none | short (3) | compare | Game Trends |
| syn_game_02 | conditional (nominal); single in practice (`fl_realised`) | none | short (1) | monitor | Game Trends |
| syn_x_01 | mixed | user, result, model | medium (8) | plan | NixOS + Game Trends |
| syn_x_02 | mixed | user, model | medium (6) | plan | NixOS + OKX |

## Axis counts
- Structure (nominal): single 2, fanout 4, chain1 1, chain2plus 1, conditional 3, mixed 7. Counting game_02 by its realised trace: single 3, conditional 2.
- Servers: 16 single-server (NixOS 8, OKX 5, Game Trends 3), 2 cross-server.
- Intent: lookup 4, compare 3, troubleshoot 3, plan 5, monitor 3.
- Length: short 8, medium 8, long 2.

## Call-weighted class estimate (realised traces, 87 calls)
Classes (from the review): F = args verbatim in user text or zero-arg; D = simple mapping from user text; R = copied from an earlier result; C = call that only exists because of an earlier result; M = model-invented/derived args; S = single-call task. One class per call; a call that is both conditional and model-invented is counted as M (or C where the argument is verbatim).

| class | calls | share |
|---|---|---|
| F | 38 | 43.7% |
| D | 11 | 12.6% |
| R | 14 | 16.1% |
| C | 7 | 8.0% |
| M | 14 | 16.1% |
| S | 3 | 3.4% |

- Strict-friendly (F+R): 52/87 = **59.8%** by calls; macro-average over tasks 52.9%. Excluding R (R is only speculation-friendly when the argument is predictable before the result arrives), F alone is 43.7%.
- Lenient-friendly (F+D+R): 63/87 = 72.4% by calls.
- Hostile (C+M+S): 24/87 = 27.6% by calls (40.2% if D counted hostile). 7/18 tasks (39%) have under 50% strict-friendly calls (okx_001, nixos_01, nixos_04, okx_01, game_02, x_01, x_02); 11/18 if tasks at exactly 50% are included.
- Zero-arg Game Trends calls: 9 of 87 (10.3%), 10 counting game_02.

Versus the review's targets: F alone is under 40%? no (43.7%, slightly over). F+R strict 45-55%? no (59.8% by calls, 52.9% macro). Zero-arg <=10%? borderline (10.3%). >=40% tasks hostile-majority? borderline (39%). The fixes cut the share from 61% to 60% and the zero-arg share from 16% to about 10%; the strict figure stays high mainly because the new chain tasks (syn_nixos_06, nixos_04) count their result-copied arguments as R. Report both the call-weighted and macro numbers, and the F-only and F+R versions.

## Per-task class breakdown (realised calls)
game_000 F6; okx_000 F3 D3; okx_001 D2 C2; nixos_000 F6 R1; nixos_01 S1; nixos_02 F2 M1 R1; nixos_03 F6; nixos_04 M4 R4 C2; nixos_05 M1 R1; nixos_06 F1 R4; nixos_07 F1 C1 M1 R1; okx_01 S1; okx_02 F5 D5; okx_03 F1 M1 R1; game_01 F3; game_02 S1; x_01 F2 M4 R1 C1; x_02 F2 C1 M2 D1.

## Grounding gaps and caveats
- NixHub, Home Manager and flakes tools errored or returned nothing at grounding time; no task depends on them. 8 of 18 NixOS tools are never exercised.
- steam_most_played and epic_trending return empty lists, so game_trends_000 (original, unchanged) has an always-empty cross-platform result, and syn_game_02's fallback would also be empty.
- okx_000 / okx_001 branch calls rarely fire on calm data; realised traces (6 and ~4 calls) are shorter than the declared workflows. Not changed per instructions.
- syn_game_01 overlap may legitimately be empty on a given day; the task has an explicit fallback.
- Index artefacts that are real tool behaviour: stable vs unstable postgresql (unstable older), stable `sqlite` = Haskell binding, `ccxt` absent, `exa` absent (so a search by the old name returns nothing, which makes syn_nixos_07 depend on the model knowing the eza rename).
- `distraction_servers` is empty for all synthetic tasks.
