# Task-set review (18 tasks: 6 MCP-Bench originals + 12 synthetic)

Reviewer method: re-derived each task's call structure from its workflow, then spot-checked with 29 real tool calls on 2026-10-01.
Findings from the spot-checks:
- Home Manager search returns nothing for `programs.neovim` and `jupyter`; `home_manager_stats` errors.
- `nixhub_find_version` returns NETWORK_ERROR.
- `nixos_flakes_search` and `nixos_flakes_stats` error.
- `darwin_search` returns nothing for `neovim` and `python`.
- `nixos_search python310` returns nothing.
- `steam_most_played` and `epic_trending` return count 0.
- `get_price MATIC-USDT` returns an error; `POL-USDT` works.
- `nixos_info terraform` on stable returns Business Source License 1.1.
- The `darwin_search dock` noise is confirmed.
- `ccxt` is not found on either channel.

## 1. Per-task verdicts

| task | verdict | reason |
|---|---|---|
| game_trends_000 | keep | Real, realistic. Calls are all zero-arg and the health gate is predictable. Caveat: `most_played` and `epic_trending` are empty, so the cross-platform result is always empty. |
| game_trends_001 | **replace** | Near-duplicate of game_000 and syn_game_01 (6 zero-arg calls). 2 of its 4 core data tools are empty. "For each of 10 titles" cannot be done with zero-arg tools. The fuzzy text spells out the whole procedure. |
| nixos_000 | **revise** | A 13-call trace, of which about 8 hit broken tools (NixHub x2, flakes x2, Home Manager search and stats, darwin_search 'neovim' empty). Several asks (0.9.2 hash, flake counts, HM option) are unanswerable. It is also the single largest source of pure fan-out calls (12 of 13). |
| nixos_001 | **replace** | Unsolvable. It needs NixHub 3.10.8, flake search and Home Manager Jupyter options, which are all broken. The darwin python/Jupyter option does not exist, and `python310` returns nothing. The realised trace is mostly errors. |
| okx_exchange_000 | keep (see fix 6) | Realistic, works. Branch calls (15m, 5m) only fire if \|dev\|>2%, which is rare on a calm day, so the realised trace is about 6 calls instead of up to 12. Fuzzy leaks thresholds (original style). |
| okx_exchange_001 | keep (see fix 6) | Same issue. >1% in 30 min and >0.5% volatility rarely fire, so the realised trace is about 3 to 4 calls. The fuzzy text is nearly step-by-step. |
| syn_nixos_01 | keep | Genuine single call. Label correct. |
| syn_nixos_02 | keep, relabel | Realistic troubleshoot. Actual structure is mixed: 2-way fan-out (channels and info), then a conditional chain (info, search, info). The label says chain2plus. Fuzzy mildly leaks ("according to the channel list"). Depends on a live license fact. |
| syn_nixos_03 | keep | Clean 6-call fan-out, user-text args. Note that the "unstable older than stable" result is partly an index artefact: the channels output shows unstable on `latest-45` and stable on `latest-46`. Harmless. |
| syn_nixos_04 | keep | Good realistic long task. Mixed label is correct (4 parallel searches, per-result infos, noisy-result fallback to prefix listing). The task_description leaks example queries, so only the fuzzy text should be shown to the agent. |
| syn_nixos_05 | keep | Clean chain1 with a model-chosen query. Top hit is query-dependent, so there is trace drift, which is acceptable. |
| syn_okx_01 | keep, relabel intent | Single call. Intent "monitor" is really lookup (a journal question). |
| syn_okx_02 | keep | 5 price calls plus 5 candle calls, all independent. Realistic. Overlaps okx_000 step 1 and nixos_03 in shape, but is the only long OKX fan-out. |
| syn_okx_03 | keep | Verified conditional with a model-invented argument (POL-USDT). Good hostile case. |
| syn_game_01 | **revise** | 2 of 4 lists return empty by construction. The fuzzy text maps 1:1 to the 4 tools and pre-announces the empty-list handling. All 4 calls are zero-arg, so they are trivially predictable. |
| syn_game_02 | keep, relabel | The conditional fallback is never taken in practice. The realised trace is a single call (`epic_free` has free games, and `epic_trending` is empty anyway). Time-sensitive: the grounded promotion ends 2026-10-01T15:00Z. |
| syn_x_01 | keep | Realistic cross-server plan, labelled mixed correctly. The package names (lutris, gamemode, mangohud, heroic) are model-inferred from vague user text, so this is a good hostile-leaning case. |
| syn_x_02 | keep, relabel | Structure is mixed: a conditional cascade plus independent OKX calls. The label says conditional. Caveat: "ccxt not packaged" is a tool-index artefact (real nixpkgs has `python3Packages.ccxt`), but behaviour is deterministic. The cascade is spelled out in the fuzzy text. |

Label re-derivation: the other `fl_*` labels (arg sources, length) are accurate. The three structure/intent mismatches are noted above (nixos_02, x_02, game_02, okx_01).
`fl_length`: short = 5 tasks, medium = 9, long = 4.

## 2. Diversity assessment (18 tasks, re-derived)

Structure, using the realised trace, with originals included:

| structure | n | tasks |
|---|---|---|
| single | 2 | nixos_01, okx_01 (game_02 is single in practice, nominally conditional) |
| fanout | 6 | game_000, game_001, nixos_000, nixos_03, okx_02, game_01 |
| chain1 | 1 | nixos_05 |
| chain2plus | 1 | nixos_001 (unsolvable, see above) |
| conditional | 2 | okx_03, game_02 |
| mixed | 6 | okx_000, okx_001, nixos_02, nixos_04, x_01, x_02 |

Other axes:
- **Argument source:** user text dominates. Model-computed args appear in about 10 calls (nixos_05, nixos_02, nixos_04, okx_03, x_01, okx_001, nixos_001). The result-derived chain is concentrated in nixos_04. There is no deep pure result-to-argument chain with working tools.
- **Servers:** 16 single-server tasks (NixOS 7, OKX 5, Game 4) and 2 cross-server (x_01, x_02). Cross-server is only 11%, and both tasks are somewhat contrived.
- **Intent:** lookup 3, compare 5, troubleshoot 2, plan 5, monitor 3. Troubleshoot is thin.

Gaps and over-representation:
- Pure chains are almost absent (1 working, 1 broken). Mixed plus fan-out make up 12 of 18 tasks, and "mixed" hides pure chain and pure conditional cases.
- Zero-arg Game Trends fan-out is over-represented: 3 near-duplicate tasks (game_000, game_001, game_01) with 16 trivially predictable calls.
- 3 tasks (game_001, game_01, game_000) and nixos_000/001 depend on broken tools.
- Hostile-structure coverage is mostly 1-call or short-conditional. Nothing has a long dependent chain.

### Bias quantification (expected calls per realised trace, 100 calls total)

Call classes:
- F = args verbatim in user text or zero-arg, independent.
- D = args need simple mapping from user text (e.g. "past week" to 1D, 7).
- R = args copied from an earlier result.
- C = conditional call (exists only if an earlier result says so).
- M = model-invented or derived args.
- S = single-call task.

| class | calls | share |
|---|---|---|
| F | 49 | 49% |
| D | 18 | 18% |
| R | 12 | 12% |
| C | 8 | 8% |
| M | 10 | 10% |
| S | 3 | 3% |

- **Friendly (strict: F+R):** 61 calls = 61% by call count. Macro-average by task: 52%.
- **Friendly (lenient: F+D+R):** 79% by call count. Macro-average: 65%.
- **Hostile (C+M+S):** 21%, or 39% if D is counted hostile. 8 of 18 tasks (44%) are hostile-majority.
- **Sensitivity to R:** the R class is only speculation-friendly when the argument is predictable before the result arrives. If R is excluded, the friendly share is F alone = 49% (strict) or F+D = 67% (lenient).
- **Where the skew comes from:** 16 zero-arg Game Trends calls (16%) and the two largest fan-outs (nixos_000 with 12 calls, okx_02 with 10) dominate by call count. About 8 calls are errors or empties from broken tools.

Judgement (no measured reference): real agent workloads are typically more 1 to 4 call sequential and single-call than this set, and fan-out of 5+ independent calls is rarer than here. The set is moderately skewed toward speculation-friendly by call count (about 60% strict, about 80% lenient), but not extremely so by task.

Recommended target:
- Strict-friendly (F+R) share of calls: 45 to 55%, in each split.
- F alone at or below 40%.
- Zero-arg calls at or below 10% of calls.
- At least 40% of tasks hostile-majority.
- Report both call-weighted and per-task macro averages.

## 3. Fix list (6 items)

1. **Replace nixos_001** with a working-tool pure chain2plus task (use `nixos_info`/`darwin_*`). Example: "Finder shows no extensions on my Mac; what else can I configure for Finder?". Calls: `darwin_search('finder')`, then `darwin_info(system.defaults.finder.AppleShowAllExtensions)`, then `darwin_options_by_prefix('system.defaults.finder')` using the prefix taken from the result, then `darwin_info` on 2 options from that list. All of these were verified working in the synthetic grounding. Gives a true 4-hop result-to-argument chain.
2. **Replace game_trends_001** with a Nix troubleshoot conditional chain: `nixos_info` on a stale or renamed package name returns NOT_FOUND, then `nixos_search` for the replacement, then `nixos_info` on the hit. Ground it with at most 5 calls first. Closes the troubleshoot gap and removes a zero-arg near-duplicate.
3. **Revise nixos_000:** drop the NixHub, flakes, Home Manager and darwin-neovim asks from both descriptions. Keep channels, search, info, stats, and a darwin option lookup that exists. Result is about 6 to 7 working calls.
4. **Revise syn_game_01:** replace `get_steam_most_played` and `get_epic_trending_games` with `get_epic_free_games` (and optionally `get_all_trending_games`). Ask for the overlap of Steam trending, Steam top sellers and Epic free. Remove the empty-list clause and make the fuzzy text stop mapping 1:1 to tools.
5. **Relabel:** nixos_02 to mixed; x_02 to mixed; okx_01 intent to lookup; game_02 to "conditional (nominal), single (realised)" and count it as single in bias stats.
6. **Make conditional branches real or record realised traces.** Either lower the okx_000 and okx_001 fuzzy thresholds (e.g. 2% to about 0.3%, 1% to about 0.05%, 0.5% volatility to a small value) so the 15m/5m/deep branches fire on typical data, or keep the originals unchanged and report realised structure and call counts alongside declared ones. Either way, cache or record tool outputs per run, since OKX and Epic data are live.

Expected effect: strict-friendly drops from 61% to about 57% of calls, broken-tool calls from about 8 to 0, zero-arg calls from 16 to 8, and pure chains rise from 1 working task to 2.

## 4. Recommended split (8 train / 10 test)

Rule:
1. Classify each task by structure (single, fanout, chain, conditional, mixed) on the post-fix roster. R1 is the replacement for nixos_001; R2 is the replacement for game_001.
2. Allocate quotas per class: single 1/1, fanout 2/3, chain 1/1, conditional 1/2, mixed 3/3 (train/test).
3. Within a class, alternate so that every server appears in both splits, the cross-server tasks are separated (x_01 train, x_02 test), and troubleshoot and monitor intents appear in both.
4. Check that strict-friendly call share is within 5 points between splits.

Train (8):
- single: syn_okx_01
- fanout: syn_nixos_03, game_trends_000
- chain: syn_nixos_05
- conditional: syn_okx_03
- mixed: okx_exchange_001, syn_nixos_04, syn_x_01

Test (10):
- single: syn_nixos_01
- fanout: syn_okx_02, nixos_000 (revised), syn_game_01 (revised)
- chain: R1 (darwin chain, replacing nixos_001)
- conditional: syn_game_02, R2 (Nix troubleshoot chain, replacing game_001)
- mixed: okx_exchange_000, syn_nixos_02, syn_x_02

Checks: train strict-friendly is about 57% (23 of 40 calls) and test is about 59% (27 of 46 calls). Every server appears in both splits. Cross-server is split 1/1. Known limitation: the only chain2plus task (R1) is in test, and train has chain1 only. Swap syn_nixos_02 into train and syn_nixos_04 into test if the train split needs deeper chains.
