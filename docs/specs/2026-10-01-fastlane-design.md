# FastLane: design

Date: 2026-10-01. Project 3 of `llm_systems_techniques/top5_projects.md`, redesigned in brainstorming.

## Question

A code-mode tool agent waits on slow tools. Speculative Programmatic Tool Calling (sPTC, Alex Zhang, Aug 2026) starts a tool call as soon as its line is complete in the streamed code. That is precise but late. A cheap predictor learned from past trajectories can start calls before the model writes anything, but it is sometimes wrong.

**Headline question:** does a free, history-learned predictor make agent turns faster on top of sPTC, and by how much at p50 and p95?

Secondary questions:
- How does the free predictor compare with the agent predicting its own next call (LLM self-prediction)?
- How much of the gain comes from prediction rather than from simply starting calls early (random-k control)?
- What do abandoned speculative calls reveal (ghost leakage), and what does a simple guard cost in hits?

Out of scope: direct (non-code) tool calling, tool retrieval, synthetic fan-out tasks, RL-trained speculators, KV-cache forking.

## Setting

- **Benchmark:** tau2-bench retail domain. tau2's environment (tools + DB), tasks and reward evaluator are used as a library. Our own orchestrator runs the loop.
- **Models:** DeepSeek Flash, thinking off, temperature 0, for both the agent and the user simulator. The agent streams.
- **Tasks:** 20 of the 36 retail `test` tasks whose gold actions write to the DB, sampled with seed 0. Read-only tasks are skipped because any episode scores 1 on them. 3 live arms per task = 60 live episodes, about $0.005 each, as measured in the smoke run.
- **History:** arm A on 20 tasks from tau2's `train` split (no overlap with test). These are the predictor's warm start.
- **Scoring:** tau2's DB-state evaluator (the final DB must match the DB after the gold actions). 39 of 40 test tasks also carry an LLM-judged natural-language assertion. We skip it to avoid judge cost and noise. Speculation never changes what the agent does, so success is only a no-harm check here.
- **Budget:** about $1.5. Hard cap $1.6 for live runs, plus a separate $0.40 cap for LLM self-prediction in replay (likely about $0.05 because the prefixes are already in DeepSeek's prompt cache; the cap covers the worst case of cache misses).
- **Tool latency:** tau2 tools are local functions, so latency is injected. Each tool gets a median drawn log-uniformly from [0.5, 5] s, seeded by tool name. Each call's latency is lognormal around that median (sigma 0.3), seeded by (task, tool, canonical args). The same call therefore gets the same latency in every arm, and in replay.
- **Read-only tools:** tools tau2 marks as READ. Only these may run speculatively. WRITE tools (cancel, modify, return, exchange, transfer) never do.

## How an episode works

1. The user simulator sends a message. The agent turn starts.
2. The agent LLM streams a reply. The reply is either text to the user, which ends the turn, or one fenced Python block.
3. The sandbox runs the script in a fresh subprocess, with no variables kept between steps. Each tool is a Python function stub. A stub sends the call to the parent over a pipe and blocks for the result. Printed output (capped at 4k characters) goes back to the agent as the next message.
4. Steps 2-3 repeat until the agent replies with text. Each episode is capped at 30 agent steps.
5. At the end, tau2's DB-state evaluator scores the episode (reward 0/1; 0 if the episode hit the step cap).

**Promise store.** It lives in the parent and maps `(tool, canonical JSON args)` to a future. Every tool call, real or speculative, goes through it:
- **Real call, key present:** wait on the existing future (a hit; it may already be done).
- **Real call, key absent:** run the call (tau2 function under a lock, then the injected sleep outside the lock) and store it.
- **Speculative call:** run it only if the key is absent and the tool is READ.
- **Any real WRITE call** clears the store. In-flight speculative futures are dropped, never used.
- **At episode end,** speculative entries that were never used count as wasted.

## Speculation sources

### sPTC (shadow execution)

The harness copies RLM and spec-ptc as closely as practical.

**The REPL (RLM LocalREPL):**
- One persistent in-process REPL per episode. Variables persist, and the tools are functions in it.
- The model writes ```` ```repl ```` blocks.
- Only `print` output comes back: a bare final expression is discarded, as in RLM.
- Output format: `REPL output:` + stdout + stderr (`ErrorType: msg`) + `REPL variables: [...]`, or `No output`, truncated at 20k characters.

**The shadow (spec-ptc):**
- **Snapshot:** each reply gets a shadow turn on a deep-copied snapshot of the REPL namespace.
- **Statements:** each closed top-level statement runs once, in order, on a shadow thread.
- **Lazy results:** a READ call goes to the promise store and returns a lazy SpecValue. The shadow waits only when a later statement reads it, so independent calls overlap (sPTC's "naive JIT").
- **Writes:** a write returns a NonSpeculated marker. Statements that read it are skipped (taint), and the shadow carries on.
- **Safety:** `print` is a no-op, risky builtins are blocked, imports are limited to a pure whitelist, and a 2 s compute guard runs per statement.
- **Peeks:** an unfinished loop runs on a throwaway copy, so its calls start early.
- **Turn end:** when the stream ends, the REPL waits for the shadow to finish, then runs the code. Real calls claim the in-flight results.
- **Eviction:** unclaimed sPTC results are evicted after each cell. Predictor guesses are not evicted.

### Trajectory predictor (counting table)

This is not ML and not an LLM: it is frequency counts plus rules.

**What it learns from** each trajectory, a sequence of tool calls with args and results grouped into user turns:
- **Next-tool counts** with back-off context: (last 2 tools), (last tool), (turn start).
- **Argument-source counts** for each (tool, arg):
  - a regex on the latest user message (email, `#W` order ID, user ID pattern), or
  - a field path in an earlier result of this episode, e.g. `get_user_details.orders[*]`.
- **Validation:** a source counts as valid only if it reproduced the real value in at least 60% of the past cases where it was available.

**Predict** (fires at turn start and after each real tool result):
1. Take next-tool probabilities from the longest context that has at least 3 observations.
2. Fill each argument from its best valid source, using this episode's values. A list source fans out to one call per element.
3. Score each candidate as p(tool) × p(source) and launch those with score ≥ τ, up to k = 3 per trigger.

**Warm start:** the predictor learns from the history episodes, our own arm-A runs on train tasks. τ is chosen on them only: learn the first 80%, then pick the smallest τ with precision ≥ 50% on the last 20%. τ is not tuned on test tasks. The public `while-ai/tau2-simulated` traces were dropped: their tool results are generic mock records, the same shape for every tool, so argument sources learned from them never match real results.

**Online learning:** after all 3 arms of a task finish, the predictor is updated with the arm-1 trajectory only. It is never updated between arms of the same task, because that would leak the task into arm 3.

## Arms

**Live**, back-to-back per task, with arm order rotated per task (ABC, BCA, CAB, ...):

| Arm | What starts early |
|---|---|
| A. Baseline | nothing |
| B. sPTC | calls whose lines have streamed |
| C. sPTC + predictor | B plus counting-table guesses |

**Replay**, free and deterministic, on recorded traces:

| Replay arm | Base trace | Purpose |
|---|---|---|
| Predictor alone | A | gain without sPTC |
| sPTC + predictor (simulated) | B | checks the simulator against live C |
| Random-k | B | same number of early calls per trigger as simulated C on that trace, picked uniformly from READ calls whose arguments are IDs already seen. Placebo and budget-matched control |
| Oracle | B | perfectly predicts the next k real calls. Ceiling |
| LLM self-prediction | A | before each agent LLM step, DeepSeek gets the exact cached prefix the agent saw plus "predict the tool calls your next code block will make, JSON, up to 3". Its own API latency delays its launches |
| Ghost guard | B + predictor | only arguments that came from the user's messages or this turn's results |
| τ sweep | B + predictor | hit vs waste curve |

### Replay simulator

The replay simulator works on one trace. A trace records, for each real call `i`:
- the request time `r_i`
- the launch time `s_i` (`= r_i` in A; the sPTC launch in B)
- the latency `L_i`
- the write calls between launches

For a new arm, the simulator does three things:
1. **Launch:** `s'_i = min(s_i, predicted launch time)`. A launch counts only if no write happened between it and the request.
2. **Wait:** `w'_i = max(0, s'_i + L_i - r'_i)`.
3. **Shift:** every later event moves earlier by the time saved so far. LLM call durations stay as recorded.

Predictor triggers use the recorded results available at that time. Speculative calls are assumed not to slow each other, since there is no contention model.

## Metrics

- **Agent-turn latency:** time from the user message to the agent's text reply. Reported at p50 and p95 over turns, plus total agent time per episode. User-simulator time is excluded.
- **Hit rate:** real READ calls served at least partly by a speculative call. Split by **seen-before vs new**: "seen" means the predictor's history had that (context, tool) transition and argument source before the call.
- **Hit@1 / Hit@3** for the predictor and for LLM self-prediction, measured before each agent LLM step that writes code: the first real call of the next script is among the top 1 / top 3 predictions, with an exact match on tool name and all args (the same rule as Speculate While You Reason).
- **Wasted tool-seconds / real tool-seconds.**
- **Ghost leakage:** abandoned speculative calls, and distinct IDs they touched that the real trajectory never used.
- **Task success** (tau2 DB-state reward) per arm, **cost per correct answer**, and **LLM spend**.
- **Simulator validity:** per-task total agent time, simulated C (from B traces) vs live C. Reported as median absolute % error.

## Statistics

- The main comparisons are C vs B and B vs A. Each is a paired per-task difference in p50/p95 turn latency and in episode agent time, with a task-clustered bootstrap CI (10,000 resamples).
- Live arms re-run the LLM, so their trajectories differ. Live comparisons are the reality check. Replay comparisons on identical traces are the precise estimate. Both are reported.
- **Order:** replay re-runs the online predictor over 3 task orders and reports the spread.
- **Mapping to the shared evaluation protocol:**
  - Placebo and budget-matched baseline: random-k.
  - Seen vs new split: the hit-rate split.
  - Orders: the 3 replay orders.
  - Cost: cost per correct answer.
  - Cached replay and caps: the error-handling rules below.

## Error handling

- **Budget:** the live run stops before starting an episode that could exceed the cap. Finished episodes are saved as trace files and skipped when the run resumes. An unfinished episode is re-run from scratch, losing at most about $0.035. Live episodes never read the LLM cache, because that would fake latency. Replay LLM calls (self-prediction) use EvoSQL's disk cache and are resumable.
- **Sandbox:** exceptions and a 120 s wall-clock timeout per script (generous, since tool sleeps count) come back to the agent as error text.
- **API:** EvoSQL's retry and backoff. A turn that needed a retry is flagged and reported, not dropped.
- **Speculative call errors** are stored like results. tau2 tool errors are deterministic, so a matching real call reuses them.

## Code layout

`fastlane/` is a standalone uv project, laid out like EvoSQL.

- `src/fastlane/llm.py`: EvoSQL's client, plus a streaming call that records chunk timestamps
- `src/fastlane/env.py`: tau2 retail wrapper (tools, READ/WRITE tags, latency injection, evaluator)
- `src/fastlane/store.py`: promise store
- `src/fastlane/sandbox.py`: script runner, shadow runner, statement splitter
- `src/fastlane/predictor.py`: counting table, warm start, τ selection
- `src/fastlane/runner.py`: episode loop, the 3 live arms, traces, budget
- `src/fastlane/replay.py`: simulator, replay arms, LLM self-prediction
- `src/fastlane/report.py`: metrics, bootstrap, plots, `summary.md`
- `configs/base.yaml`, `tests/`

## Tests

Tests cover the core logic only:
- **Store:** a hit on an in-flight call waits only the remaining time, a write clears the store, and duplicates are removed.
- **Splitter:** a `for` block is incomplete until it ends, and a partial line is not a statement.
- **Shadow:** a dependent call chains, and a WRITE stops the shadow.
- **Predictor:** back-off order, the source-validation threshold, fan-out, and the τ gate.
- **Simulator:** hand-built traces with known answers for a fully hidden call, a partial overlap, write invalidation, and the time shift.
- **tau2 parity:** a recorded trajectory gets the same reward through our loop as through tau2's evaluator.

## Expected result and risks

- **Expected:** 30-45% hit rate for the predictor and a 5-15% p50 cut vs baseline. That's in line with PASTE's top-3 recall of 43.9% and toolspec's 11.5%. The gain on top of sPTC may be small, because with thinking off the window before a call line streams is about 1-3 s. A small but honestly measured gain is still a valid result.
- **Risk: live noise.** With 14 tasks, DeepSeek's latency noise may hide the live differences. That's why replay is the precise estimate.
- **Risk: warm-start mismatch.** The public traces use direct calls, so their trajectory shape differs from code mode. Online updates partly correct this.
- **Downloads** (need user approval with URL and size first): the tau2-bench source archive at a pinned commit (installed as a package), five retail data files from the tau2 repo (about 3.2 MB), and `while-ai/tau2-simulated` `data/retail/train.jsonl` (8.9 MB).

## Prior work

- sPTC: alexzhang13.github.io/blog/2026/spec-ptc, github.com/alexzhang13/spec-ptc
- PASTE (pattern-aware speculative tool execution): arXiv 2603.18897
- toolspec (n-gram speculation): github.com/joelvarun/toolspec
- Speculate While You Reason (τ-bench Hit@1): arXiv 2607.25816
- SPORK (self-speculative forking): arXiv 2607.03333
- Ghost Tool Calls: arXiv 2606.02483
