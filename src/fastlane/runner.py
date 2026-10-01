"""Live episodes on tau2 retail: an RLM-style agent (one persistent REPL per episode) with no speculation (A),
sPTC (B) or sPTC + predictor (C). Each episode becomes a trace with every timestamp the replay needs (seconds from start)."""
import json
import random
import time
from itertools import permutations
from pathlib import Path

from fastlane import env as tau
from fastlane.budget import Budget, BudgetExceeded
from fastlane.common import call_latency, parse_json
from fastlane.files import write_atomic
from fastlane.llm import make_llm
from fastlane.predictor import State, from_trace, load_history, warm_predictor
from fastlane.repl import MAX_OUTPUT, Repl, ShadowTurn, code_blocks, format_output
from fastlane.store import Store

ORDERS = ["".join(p) for p in permutations("ABC")]  # all 6 arm orders, balanced across tasks
GREETING = "Hi! How can I help you today?"
MAX_STEPS, MAX_TURNS = 30, 25
AGENT_PROMPT = """You are a customer service agent for an online retail store. Follow the policy exactly.

<policy>
{policy}
</policy>

You act through a persistent Python REPL. To use tools, reply with one or more ```repl code blocks and nothing else.
The tools below are functions in the REPL. They return parsed JSON (dict, list or str) and raise ToolError on failure.
Variables persist between code blocks, so store results in variables and reuse them instead of calling a tool again.
The REPL is NOT a Jupyter cell: only print(...) output (stdout) is shown back to you, and a bare expression on the
last line is silently discarded. Always wrap inspections in print(...).
Only pure standard-library imports (json, re, math, collections, itertools, ...) work; there is no file or network access except through the tools.
To talk to the customer, reply with plain text and no code block.

Tools:
{tools}"""


def run_step(messages, arm, agent_llm, store, repl, writes, clock, sink):
    """One streamed reply run in the episode REPL; in B/C an sPTC shadow turn runs during the stream, and the REPL waits for it."""
    step = {"n_messages": len(messages), "t_start": clock(), "calls": [], "code": None, "output": None}
    sink["calls"] = step["calls"]
    turn = ShadowTurn(repl, lambda tool, args: store.speculate(tool, args, "sptc"), writes) if arm in ("B", "C") else None
    first_log = len(store.log)

    def on_text(text):
        blocks = code_blocks(text)
        if turn and blocks:
            turn.feed("\n".join(blocks))
    reply = agent_llm.stream(messages, on_text)
    if turn:
        turn.end()
    step["llm"] = {k: reply[k] for k in ("content", "chunks", "t_first", "latency_s", "retried")}
    messages.append({"role": "assistant", "content": reply["content"]})
    blocks = code_blocks(reply["content"])
    if blocks:
        step["code"] = "\n".join(blocks)
        out = "\n\n".join(format_output(*repl.run(b), repl.ns) for b in blocks)
        step["output"] = out if len(out) <= MAX_OUTPUT else out[:MAX_OUTPUT] + f"... + [{len(out) - MAX_OUTPUT} chars...]"  # cap per reply
        messages.append({"role": "user", "content": step["output"]})
        store.evict("sptc")
    step["sptc"] = [{"tool": e["tool"], "args": e["args"], "offset": round(e["t_launch"] - step["t_start"], 3)}
                    for e in store.log[first_log:] if e["source"] == "sptc"]
    step["t_end"] = clock()
    return step


def run_episode(task, arm, cfg, agent_llm, user_llm, predictor=None):
    """One live episode -> trace dict."""
    retail, t0 = tau.RetailEnv(), time.monotonic()

    def clock():
        return round(time.monotonic() - t0, 3)
    store = Store(retail.execute, lambda tool, args: call_latency(task.id, tool, args, cfg["latency"]), retail.read, clock)
    specs, writes, state, sink = retail.tool_specs(), set(retail.fns) - retail.read, State(), {}

    def call(tool, args):
        content, error, info = store.call(tool, args)
        sink["calls"].append(info)
        return content, error
    repl = Repl(specs, call)

    def launch():
        if arm == "C":
            for tool, args, _ in predictor.predict(state):
                store.speculate(tool, args, "pred")

    def on_result(info):
        state.add_call(info["tool"], parse_json(info["content"]))
        launch()
    store.on_result = on_result
    user = [{"role": "system", "content": tau.user_system_prompt(task)}, {"role": "user", "content": GREETING}]
    messages = [{"role": "system", "content": AGENT_PROMPT.format(policy=retail.env.get_policy(), tools=retail.tool_docs())},
                {"role": "assistant", "content": GREETING}]
    turns, steps, termination = [], 0, "max_steps"
    while len(turns) < MAX_TURNS:
        user_text = user_llm.chat(user)["content"] or ""
        user.append({"role": "assistant", "content": user_text})
        turn = {"user": user_text, "t_start": clock(), "steps": [], "reply": None, "retried": False}
        turns.append(turn)
        if any(s in user_text for s in tau.STOP_TOKENS):
            termination, turn["t_end"] = "user_stop", turn["t_start"]
            break
        messages.append({"role": "user", "content": user_text})
        state.start_turn(user_text)
        launch()
        while steps < MAX_STEPS and turn["reply"] is None:
            steps += 1
            step = run_step(messages, arm, agent_llm, store, repl, writes, clock, sink)
            turn["steps"].append(step)
            turn["retried"] |= step["llm"]["retried"]
            if step["code"] is None:
                turn["reply"] = step["llm"]["content"]
        turn["t_end"] = clock()
        if turn["reply"] is None:
            break
        user.append({"role": "user", "content": turn["reply"]})
    store.pool.shutdown(wait=False, cancel_futures=True)
    return {"task_id": task.id, "arm": arm, "termination": termination,
            "reward": tau.score(task, tau.to_tau2(turns, GREETING), termination),
            "read_tools": sorted(retail.read), "tool_specs": specs, "latency_cfg": cfg["latency"], "turns": turns,
            "spec_log": [{k: e[k] for k in ("tool", "args", "source", "t_launch", "used")} for e in store.log],
            "agent_messages": messages}


def run_and_save(path, task, arm, cfg, agent, user, budget, predictor=None):
    """One episode saved to path (skipped if it exists); False means stop for budget."""
    if path.exists():
        return True
    if budget.total + cfg["budget"]["episode_usd"] > budget.cap:
        print(f"stopping: ${budget.total:.3f} spent of ${budget.cap:.2f}; raise the cap and rerun to resume")
        return False
    try:
        trace = run_episode(task, arm, cfg, agent, user, predictor)
    except BudgetExceeded as e:
        print(f"stopping mid-episode ({e}); that episode reruns from scratch next time")
        return False
    write_atomic(path, json.dumps(trace))
    print(f"task {task.id} arm {arm}: reward {trace['reward']}, {len(trace['turns'])} turns, ${budget.total:.3f} spent")
    return True


def setup(cfg):
    runs = Path(cfg["paths"]["runs"])
    budget = Budget(runs / "spend.json", cfg["budget"]["live_usd"])
    return runs, budget, *(make_llm(cfg["agent"], None, on_spend=budget.spend) for _ in range(2))


def history(cfg, limit=None):
    """Arm A on train-split tasks: the predictor's warm-start history (no overlap with the test tasks)."""
    runs, budget, agent, user = setup(cfg)
    tasks = tau.sample_tasks(**cfg["history"])
    write_atomic(runs / "history" / "order.json", json.dumps([t.id for t in tasks]))
    for task in tasks[:limit]:
        if not run_and_save(runs / "history" / f"{task.id}.json", task, "A", cfg, agent, user, budget):
            return


def record(cfg, limit=None):
    """Live run: fixed task order, arm order rotated per task; finished episodes are skipped on resume."""
    runs, budget, agent, user = setup(cfg)
    tasks = tau.sample_tasks(**cfg["tasks"])
    write_atomic(runs / "order.json", json.dumps([t.id for t in tasks]))
    tau_path, n_hist = runs / "tau.json", len(load_history(runs))
    predictor = warm_predictor(cfg, tau.RetailEnv().read,
                               json.loads(tau_path.read_text())["tau"] if tau_path.exists() else None)
    if n_hist >= cfg["history"]["n"]:
        write_atomic(tau_path, json.dumps({"tau": predictor.tau}))
    else:
        print(f"warning: history has {n_hist} of {cfg['history']['n']} episodes; tau={predictor.tau} is not saved")
    orders = random.Random(cfg["tasks"]["seed"]).sample(ORDERS, 6) * (len(tasks) // 6 + 1)
    for task, order in list(zip(tasks, orders))[:limit]:
        for arm in order:
            if not run_and_save(runs / arm / f"{task.id}.json", task, arm, cfg, agent, user, budget, predictor):
                return
        predictor.learn(from_trace(json.loads((runs / "A" / f"{task.id}.json").read_text())))
