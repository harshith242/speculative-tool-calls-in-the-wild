"""MCP-Bench episodes: one task instruction, the agent works in the RLM REPL until it answers in plain text.
Arms: A baseline, B sPTC, C sPTC + predictor, P predictor without sPTC (live replay check)."""
import json
import random
import time
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

from fastlane.budget import Budget, BudgetExceeded
from fastlane.common import parse_json
from fastlane.files import write_atomic
from fastlane.llm import make_llm
from fastlane.mcp_env import MCPEnv
from fastlane.predictor import State, from_trace
from fastlane.repl import Repl
from fastlane.runner import run_step
from fastlane.store import Store

AGENT_PROMPT = """You are a capable research assistant. Solve the user's task with the tools below.

You act through a persistent Python REPL. To use tools, reply with one or more ```repl code blocks and nothing else.
The tools below are functions in the REPL; they return parsed JSON (dict, list or str) or text, and raise ToolError on failure.
Variables persist between code blocks, so store results and reuse them instead of calling a tool again.
The REPL is NOT a Jupyter cell: only print(...) output is shown back to you; a bare last expression is discarded.
Only pure standard-library imports (json, re, math, collections, itertools, ...) work; there is no file or network access except through the tools.
Print only what you need (outputs over 20k characters are truncated). When done, reply with the final answer in plain text.

Tools:
{tools}"""


def load_tasks(cfg):
    """{task_id: {"text": fuzzy_description, "servers": [...]}} for every task in the split."""
    out = {}
    for f in sorted(Path(cfg["tasks_dir"]).glob("*.json")):
        for group in json.loads(f.read_text())["server_tasks"]:
            servers = group["server_name"].split("+")  # "servers" is a stringified list in the files
            for t in group["tasks"]:
                out[t["task_id"]] = {"text": t["fuzzy_description"], "servers": servers}
    return {t: out[t] for t in cfg["split"]["train"] + cfg["split"]["test"]}


def run_mcp_episode(task_id, task, arm, cfg, agent_llm, predictor=None):
    """One live episode -> trace (same format as v1; calls carry measured durations)."""
    root = cfg["root"]
    servers = {s: {**cfg["servers"][s], "cwd": f"{root}/{cfg['servers'][s]['cwd']}"} for s in task["servers"]}
    env, t0 = MCPEnv(list(servers), servers, cfg["safe"], python=cfg["python"]), time.monotonic()
    try:
        def clock():
            return round(time.monotonic() - t0, 3)
        store = Store(env.execute, None, env.read, clock, cap=cfg["caps"]["predictor_per_episode"])
        specs, writes, state, sink = env.tool_specs(), set(env.tools) - env.read, State(), {}

        def call(tool, args):
            content, error, info = store.call(tool, args)
            sink["calls"].append(info)
            return content, error
        repl = Repl(specs, call)

        def launch():
            if arm in ("C", "P"):
                for tool, args, _ in predictor.predict(state):
                    store.speculate(tool, args, "pred")

        def on_result(info):
            state.add_call(info["tool"], parse_json(info["content"]))
            launch()
        store.on_result = on_result
        messages = [{"role": "system", "content": AGENT_PROMPT.format(tools=env.tool_docs())},
                    {"role": "user", "content": task["text"]}]
        turn = {"user": task["text"], "t_start": clock(), "steps": [], "reply": None, "retried": False}
        state.start_turn(task["text"])
        launch()
        step_arm = "A" if arm == "P" else arm  # P speculates by prediction only, no shadow
        while len(turn["steps"]) < cfg["caps"]["max_steps"] and turn["reply"] is None:
            step = run_step(messages, step_arm, agent_llm, store, repl, writes, clock, sink)
            turn["steps"].append(step)
            turn["retried"] |= step["llm"]["retried"]
            if step["code"] is None:
                turn["reply"] = step["llm"]["content"]
        turn["t_end"] = clock()
        store.pool.shutdown(wait=False, cancel_futures=True)
        return {"task_id": task_id, "arm": arm, "servers": task["servers"], "termination": "answered" if turn["reply"] else "max_steps",
                "read_tools": sorted(env.read), "tool_specs": specs, "latency_cfg": None, "turns": [turn],
                "spec_log": [{k: e.get(k) for k in ("tool", "args", "source", "t_launch", "t_end", "used", "failure")} for e in store.log],
                "agent_messages": messages}
    finally:
        env.close()


def deepseek_peak(now=None):
    """DeepSeek's weekday peak hours (01-04 and 06-10 UTC), when prices are double the off-peak rates we budget with."""
    now = now or datetime.now(timezone.utc)
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def episode(runs, task_id, task, arm, cfg, agent, budget, predictor):
    """One episode saved under runs/arm (skipped if it exists); False means stop (budget or peak hours)."""
    path = runs / arm / f"{task_id}.json"
    if path.exists():
        return True
    if deepseek_peak():
        print("stopping: DeepSeek peak hours (prices are budgeted at off-peak rates); rerun after 10:00 UTC / 3:30 pm IST")
        return False
    if budget.total + cfg["budget"]["episode_usd"] > budget.cap:
        print(f"stopping: ${budget.total:.3f} of ${budget.cap:.2f}")
        return False
    try:
        trace = run_mcp_episode(task_id, task, arm, cfg, agent, predictor)
    except BudgetExceeded as e:
        print(f"stopping mid-episode ({e})")
        return False
    write_atomic(path, json.dumps(trace))
    calls = sum(len(s["calls"]) for s in trace["turns"][0]["steps"])
    print(f"{task_id} {arm}: {calls} calls, {trace['turns'][0]['t_end']:.1f}s, ${budget.total:.3f}")
    return True


def mcp_run(cfg, phase, limit=None):
    """'baseline': arm A on all split tasks. 'live': cfg live_arms (default ABC) on test with balanced arm orders, C's predictor
    learning online; P (predictor, no sPTC, frozen train predictor) on 6 test tasks if listed."""
    from fastlane.replay import train_predictor
    runs = Path(cfg["paths"]["runs"]) / phase
    budget = Budget(runs / "spend.json", cfg["budget"]["baseline_usd" if phase == "baseline" else "live_usd"])
    agent, tasks = make_llm(cfg["agent"], None, on_spend=budget.spend), load_tasks(cfg)
    if phase == "baseline":
        for task_id in list(tasks)[:limit]:
            if not episode(runs, task_id, tasks[task_id], "A", cfg, agent, budget, None):
                return
        return
    arms = cfg.get("live_arms", "ABC")
    pool = cfg["split"]["test"] + (cfg["split"]["train"] if cfg.get("live_tasks") == "all" else [])  # sPTC learns nothing
    test = pool[:limit]
    online = train_predictor(cfg) if "C" in arms or "P" in arms else None
    perms = ["".join(p) for p in permutations(arms.replace("P", ""))]
    orders = random.Random(0).sample(perms * -(-len(pool) // len(perms)), len(pool))
    for task_id, order in zip(test, orders):
        for arm in order:
            if not episode(runs, task_id, tasks[task_id], arm, cfg, agent, budget, online):
                return
        if online:
            online.learn(from_trace(json.loads((runs / "A" / f"{task_id}.json").read_text())))
    if "P" in arms:
        frozen = train_predictor(cfg, tau=online.tau)
        for task_id in test[:6]:
            if not episode(runs, task_id, tasks[task_id], "P", cfg, agent, budget, frozen):
                return


def calibrate_command(cfg):
    """Re-issue up to 3 recorded safe calls per tool at concurrency 1/2/4/8 (no LLM); then write the structure report."""
    from fastlane.structure import calibrate, write_structure
    runs, out = Path(cfg["paths"]["runs"]) / "baseline" / "A", Path(cfg["paths"]["results"])
    calls, seen = {}, set()
    for t in cfg["split"]["train"]:
        if (runs / f"{t}.json").exists():
            trace = json.loads((runs / f"{t}.json").read_text())
            for c in (c for s in trace["turns"][0]["steps"] for c in s["calls"] if c["tool"] in trace["read_tools"]):
                key = (c["tool"], json.dumps(c["args"], sort_keys=True))
                if key not in seen and len(calls.setdefault(c["tool"], [])) < 3:
                    seen.add(key)
                    calls[c["tool"]].append((c["tool"], c["args"]))
    root = cfg["root"]
    servers = {s: {**v, "cwd": f"{root}/{v['cwd']}"} for s, v in cfg["servers"].items()}
    env = MCPEnv(list(servers), servers, cfg["safe"], python=cfg["python"])
    try:
        factors = calibrate(env.execute, [c for cs in calls.values() for c in cs])
    finally:
        env.close()
    write_atomic(out / "calibration.json", json.dumps(factors, indent=1))
    write_structure(cfg, out / "structure.md")
    print(f"wrote {out}/calibration.json and {out}/structure.md")
