# FastLane v2: design (MCP-Bench, real tools)

Date: 2026-10-01. This follows v1 on tau2 retail and an external review of the v2 plan. Budget: about $0.90 of LLM spend in total, with hard caps.

## Scope after the runs (decided 2026-10-01, before the final n=18 live run)

**sPTC is the headline.** The live experiment is A (baseline) vs B (sPTC), measured on all 18 tasks. sPTC learns nothing, so it needs no train/test split.

**Arm C (counting-table predictor) is an appendix**, reported from replay only. With the pre-registered rule (τ by leave-one-task-out on train), τ = 0.8, which switches the predictor off: on 8 diverse train tasks, no lower τ saved more time than it wasted. Live C would equal B, and P would equal A, so neither was run.

**Disclosure (forking paths).** The predictor gained typed constant arguments, prose list items and generic user-text patterns, plus a read-set bug fix, *after* we saw it never fire in test-set replays. The controls were fixed at the same time. These changes are generic, not task-specific, and τ stayed train-tuned. The outcome is a null result for C, so nothing was inflated, but the sequence is stated here.

## Why v2

v1 used tau2 retail with injected tool delays. Its gains were real for that setup but inflated: the workflow is very repetitive (8 call shapes in 20 tasks), the delays were invented, and random prefetching was nearly as fast as the predictor.

v2 asks the question that matters for real agents:

**How much wall-clock latency do sPTC and the counting-table predictor each remove when a code-mode agent uses real tools with real network latency on diverse workflows, and is the predictor's share due to *prediction* rather than just starting calls early?**

## Claim we will test (narrowed per review)

FastLane reduces latency by exploiting critical-path overlap in programmatic tool-use trajectories:
- sPTC hides calls whose code has already been written, and runs independent calls in parallel.
- The predictor adds benefit only where the next call's tool and arguments can be inferred from the user's text or from earlier tool results.

## Benchmark and task pool

**MCP-Bench** (Accenture, ICLR 2026): 28 live MCP servers and 104 tasks. Each task has a natural-language prompt (`fuzzy_description`) and the authors' workflow notes (`dependency_analysis`). We use only the natural-language prompt as the agent's input.

**Pool: 3 servers (OKX Exchange, Game Trends, NixOS), 18 frozen tasks.**

**How the pool was chosen.** Every key-free networked MCP-Bench server was checked, and most failed:
- **Rate limits:**
  - Wikipedia refused 65% of anonymous requests (HTTP 429).
  - DexPaprika allows 30 requests per minute.
  - Met Museum refuses bursts (403).
  - Paper Search is throttled 8–32× under concurrency (arXiv, PubMed).
- **Serial handling:** Call for Papers and FruityVice handle calls one at a time (11–17× slower with 16 concurrent calls), so nothing can overlap.
- **Security:** Context7 has CVE-2026-75130 (CVSS 9.0, prompt injection into the agent's context), and MCP-Bench ships v1.0.0. OpenAPI Explorer has 2 critical/high AgentSeal findings and failed 13 of 16 lookups.
- **Keys or ethics:** Weather and Movie Recommender actually need keys; Car Price allows about 500 requests/day; OSINT fires lookups at third-party hosts.

The 3 kept servers had zero refusals at 16 concurrent calls and safe security reviews (AgentSeal OKX 98.6, NixOS 98.3; Game Trends is a few plain GETs).

**Task set: 4 MCP-Bench originals + 14 synthetic** (`tasks_v2/`).
- **Originals:** okx_exchange_000/001 and game_trends_000 are kept unchanged. nixos_000 was revised to drop the asks that need broken tools. nixos_001 and game_trends_001 were replaced, because they depended on tools that are currently broken (NixHub, flakes, Home Manager, empty Game Trends lists).
- **Synthetic tasks:** written by one agent and grounded with real tool calls. They follow a diversity table fixed in advance:
  - structure: single / fanout / chain1 / chain2plus / conditional / mixed
  - argument source: user text / earlier result / model-computed
  - servers: single vs cross-server
  - length
  - intent
- **Review:** an independent reviewer agent audited realism, solvability, label accuracy, workflow leakage and bias toward speculation (`tasks_v2/review.md`). One revision round fixed its 6 findings.
- **Final mix:** single 2, fanout 4, chain1 1, chain2plus 1, conditional 3, mixed 7; 16 single-server tasks (NixOS 8, OKX 5, Game Trends 3) and 2 cross-server tasks.
- **Speculation-friendly calls** (args in the user's text with independent calls, or args copied from earlier results): **53% per task** (inside the reviewer's 45–55% target), **60% call-weighted** (above it). Both numbers are reported.
- **Agent input:** only `fuzzy_description` reaches the agent. The workflow fields are for analysis only.

**Install safety (`scripts/install_mcp_servers.sh`):**
- Sparse checkout of the 3 server folders only.
- OKX: `npm` with `--ignore-scripts`, `axios` pinned to 1.20.0 (0 OSV records, GitHub Actions provenance), built with its own TypeScript.
- Python servers: separate uv envs with `exclude-newer` (NixOS needs the v1 MCP SDK).
- Before any run: OSV check of all lockfiles and env packages (no malware found), plus a scan for `.pth` files.
- The agent's REPL is hardened (no files, no `eval`/`exec`, pure imports only), because tool output is untrusted.

### Train / test split (fixed before any run)

The rule is the reviewer's stratified rule:
- per-structure quotas (train/test): single 1/1, fanout 2/3, chain 1/1, conditional 1/2, mixed 3/3;
- every server appears in both splits;
- the 2 cross-server tasks are separated;
- the speculation-friendly share stays within about 5 points between splits.

| | Tasks |
|---|---|
| Train / history (8) | syn_okx_01, syn_nixos_03, game_trends_000, syn_nixos_05, syn_okx_03, okx_exchange_001, syn_nixos_04, syn_x_01 |
| Test (10) | syn_nixos_01, syn_okx_02, nixos_000, syn_game_01, syn_nixos_06, syn_game_02, syn_nixos_07, okx_exchange_000, syn_nixos_02, syn_x_02 |

**What the test set measures:**
- All test tasks run on servers seen in train, so "unseen" generalization is measured per transition (seen-before vs new calls), not per server.
- The only deep chain (syn_nixos_06) is in test.

**Learning rules:**
- **τ:** chosen by leave-one-task-out on the 8 train tasks. The τ curve on test is reported, never used to choose.
- **Online learning:** the predictor also learns online from test tasks already finished. Results are reported both frozen and online.

**Known bias:** see the speculation-friendly share above. Results are also reported per structure class.

## Harness

The harness is the same as tau2 v1, adapted:
- **REPL:** RLM-style persistent REPL.
- **sPTC:** the faithful spec-ptc shadow.
- **Store:** the shared promise store.
- **Predictor:** the counting-table predictor.
- **Model:** DeepSeek Flash with thinking off.
- **Episode:** single-instruction. System prompt + task prompt, then agent steps until a plain-text answer. At most 20 steps. No user simulator.
- **MCP client:** one stdio session per server on a background asyncio loop. Tools become REPL functions. Calls from store threads use `run_coroutine_threadsafe`.
- **Real latency:** no injected delay. Each call's measured duration (`t_done − t_launch`) is recorded.
- **Grading:** not needed. Speculation never changes agent behaviour, so latency is measured without grading. We keep MCP-Bench's prompts but not its LLM judge.

## Safety: `speculation_safe` whitelist (replaces `readOnlyHint`)

A tool may be started early only if it is on a hand-checked whitelist. We read each tool's implementation and confirm it is:
- read-only
- safe to call twice
- free of quota or billing cost beyond public rate limits
- free of private arguments
- tolerant of a few concurrent calls

`readOnlyHint` is recorded but never trusted. Missing annotations are not "fixed" by assumption.

**Caps:** the predictor (and every control) starts at most 3 calls per trigger and 12 per episode. sPTC launches are not capped: they are calls the agent has already written and will make anyway, so they add no extra load.

**Failure logging:** speculative failures are logged by kind (timeout, rate limit, transport error, argument mismatch, abandoned).

## Phases

### Phase 0: baseline and structure (live, arm A only)
1. **Calibration:** call each whitelisted tool at concurrency 1, 2, 4 and 8 with benign arguments, about 400 calls with no LLM. This gives a per-tool slowdown factor under concurrency.
2. **Baseline:** run all 18 tasks once. Pilot 2 tasks first to measure cost per episode. The train baselines become the predictor's history, and the test baselines feed the replay.
3. **Structure analysis** of the baseline traces (free):
   - **Overlap opportunity per call:** O_i = min(D_i, t_request − t_earliest_safe). The earliest safe time is:
     - turn start, for arguments found in the task text;
     - completion of the producing call, for arguments found in an earlier result;
     - the end of the model's stream, for model-made arguments.
   - **Overlap ceiling:** ΣO / episode time.
   - **Argument-source mix:** user text, earlier result field, or model-made.
   - **Fan-out:** the number of calls that could run at the same time.
   - **Chain depth.**
   - **Cross-server hops.**
   - **Structure classes:** F0/F1/F2 (fan-out), D0/D1/D2 (dependent hops), X (cross-server), B (branching).
4. **Test set:** fixed by the split above, not chosen after seeing results. The structure analysis only describes the train and test sets.

### Phase 1: replay on baseline traces (free)
These arms re-time identical baseline conversations with measured per-call durations and the calibrated concurrency slowdown. There is no sPTC here (see the v1 lesson: a faithful sPTC joins the shadow before execution, so B/C traces cannot be re-timed).

**Placebo ladder** (all launch at the same triggers, with the same per-trigger budget as the predictor):
1. Random tool: random safe tool with arguments from known IDs.
2. Frequency prior: next tool by plain transition frequency, arguments from known IDs at random.
3. Right tool, wrong argument: the predictor's tool with a deliberately different known value.
4. Learned predictor.
5. Budgeted exhaustive prefetch: every safe call formable from known values, K = 2/4/8.
6. Oracle: the exact next k calls.

**Other replay analyses:**
- **Ghost guard:** only allow arguments from this turn's text or results.
- **LLM self-prediction:** the agent predicts its own next calls (paid, about $0.03).
- **History learning curve:** warm-start on 0, 5 and all history episodes. Hits are split into unseen server / seen server / seen transition.
- **τ curve:** latency saved, wasted tool-seconds and ghost calls across τ. τ is picked by latency saved − λ·wasted.
- **Delay-ratio curve:** speed-up vs (tool latency ÷ LLM generation time), made by scaling measured durations.

### Phase 2: live A/B/C (the sPTC and predictor result)
- **Episodes:** 10 test tasks × arms A, B, C, run back-to-back per task, with the arm order drawn from the 6 orders, plus P on 6 test tasks. Together with 18 baselines, that's 54 episodes, about $0.65 at off-peak prices.
- **Predictor:** trained on the 10 train baselines, with τ chosen by leave-one-task-out on train. It runs online (learns from earlier test tasks); the replay also reports the frozen variant.
- **Live check:** a "predictor without sPTC" arm on 6 of the tasks validates the replay counterfactual, with replay error reported per task.

## Metrics

**Primary endpoint (pre-registered):** agent time per episode, as a geometric-mean ratio over tasks. The comparisons are B/A (sPTC) and C/B (predictor on top of sPTC), live. Each gets a 95% bootstrap CI over tasks and a two-sided sign-flip permutation p.

**Replay primary:** learned predictor / frequency prior, and learned predictor / random tool. This answers whether prediction quality matters beyond simply starting calls early.

**Secondary metrics:**
- p50 and p90 turn latency (p95 is not used with n = 12)
- hit rate split by argument source and by seen/new
- wasted tool-seconds per second saved
- call amplification: all calls ÷ necessary calls
- speculative failure rate by kind
- distinct speculative entities (ghost exposure)
- Hit@1/Hit@3 for the predictor vs self-prediction
- overlap ceiling captured: saved ÷ ceiling
- results per structure class

## What would falsify a positive C result (from the review)

Any of these would count against it:
- Live C fails to beat B while replay says it should.
- The frequency prior or the right-tool/wrong-argument control nearly matches the learned predictor.
- The gain sits in 2–3 tasks.
- C raises the speculative failure or rate-limit rate.
- C's trajectories differ from A/B on identical tool outputs (a harness bug).
- The gain tracks the number of speculative calls rather than hit quality.
- The gain appears only on seen transitions.

## Budget and caps

| Item | Estimate | Cap |
|---|---|---|
| Phase 0 baseline (pilot 2, then 20 more) | ~$0.20 | $0.30 |
| Phase 2 live (36 episodes + 6 predictor-only) | ~$0.40 | $0.50 |
| Self-prediction replay | ~$0.03 | $0.10 |
| **Total** | **~$0.65** | **$0.90** |

**Runs:** every run is resumable and stops before an episode that could exceed its cap. Tool outputs are capped at 20k characters per block (RLM), which keeps prompt sizes bounded.

## Out of scope / future work
- A second benchmark (MCP-Atlas, which has many key-requiring servers; or MCPMark, which needs sandboxed accounts).
- Mixed-effects models (n is too small).
- Hashing logged arguments (benchmark data is public, but production use would need it).
- Servers that need keys.
