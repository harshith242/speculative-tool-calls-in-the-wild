"""Replay: re-time recorded episodes on a simulated clock under other speculation sources (free, deterministic).
LLM durations and script gaps stay as recorded; each tool wait becomes max(0, launch + duration - request).
v2 (MCP): measured call durations, concurrency slowdown, control ladder, curves."""
import copy
import json
import math
import random
import re
import statistics
from pathlib import Path

from fastlane.budget import Budget, BudgetExceeded
from fastlane.common import call_latency, canonical, parse_json
from fastlane.files import write_atomic
from fastlane.llm import extract_json, make_llm
from fastlane.predictor import USER_PATTERNS, Predictor, State, from_trace, walk, warm_predictor

ID_PATTERNS = {"email": USER_PATTERNS["email"], "order_id": r"#W\d{7}", "user_id": USER_PATTERNS["user_id"],
               "product_id": r"\b\d{10}\b", "item_id": r"\b\d{10}\b"}
SELF_PROMPT = ("Before you answer: predict the tool calls your next code block will make, most likely first. "
               'Reply only with JSON {"calls": [{"tool": "<name>", "args": {...}}]} with at most 3 calls, '
               'or {"calls": []} if you will reply in plain text.')


class KnownState(State):
    """State plus every entity string seen so far: result leaves and the task text's quoted / Capitalised phrases."""

    def __init__(self):
        super().__init__()
        self.known = []

    def note(self, values):
        self.known += [v for v in dict.fromkeys(values) if 2 <= len(v) <= 80 and v not in self.known]

    def start_turn(self, user):
        super().start_turn(user)
        found = re.findall(r'"([^"]+)"|(?<!\w)\'([^\']+)\'(?!\w)|([A-Z][a-z]+(?: [A-Z][a-z]+)*)|\b([A-Z0-9]{2,}(?:-[A-Z0-9]{2,})+)\b', user)
        self.note(g for m in found for g in m if g)

    def add_call(self, tool, result):
        super().add_call(tool, result)
        self.note(v for _, v in walk(result))


def call_duration(trace, c):
    """Measured seconds of a recorded call (v1 traces: the seeded latency model)."""
    if c.get("duration") is not None:
        return c["duration"]
    return call_latency(trace["task_id"], c["tool"], c["args"], trace["latency_cfg"])


def simulate(trace, launcher=None, use_sptc=False, seen=None, slowdown=None, scale=1.0):
    """Re-time one trace. launcher(event) -> [(tool, args, delay)] at each trigger; use_sptc replays the shadow's launches.
    slowdown {tool: {level: factor}} stretches a launch by the factor of its in-flight launch count (rounded down to a level)."""
    read = set(trace["read_tools"])
    table, launches, calls, turn_times, triggers = {}, [], [], [], []
    state, T = KnownState(), 0.0
    real, by_tool = {}, {}
    for turn in trace["turns"]:
        for s in turn["steps"]:
            for c in s["calls"]:
                real.setdefault((c["tool"], canonical(c["args"])), call_duration(trace, c))
                by_tool.setdefault(c["tool"], []).append(call_duration(trace, c))
    every = [d for ds in by_tool.values() for d in ds]

    def guess_duration(tool, key):
        """A real call's duration; a wrong guess (never made for real) gets its tool's median, else the overall median."""
        if key in real:
            return real[key]
        return statistics.median(by_tool.get(tool) or every or [0.0])

    def factor(tool, inflight):
        levels = {int(k): f for k, f in (slowdown or {}).get(tool, {}).items() if int(k) <= inflight}
        return levels[max(levels)] if levels else 1.0

    def launch_all(items, source):
        new = []
        for tool, args, t in items:
            key = (tool, canonical(args))
            if tool in read and key not in table:
                table[key] = {"tool": tool, "args": args, "t": t, "source": source, "used": False, "dur": 0.0, "end": math.inf}
                launches.append(table[key])
                new.append((key, table[key]))
            elif tool in read and t < table[key]["t"]:
                table[key].update(t=t, end=t + table[key]["dur"], source=source)  # the earliest launch wins (a delayed guess can be overtaken)
        for key, rec in new:
            inflight = 1 + sum(1 for r in launches if r is not rec and r["t"] <= rec["t"] < r["end"])
            rec["dur"] = guess_duration(rec["tool"], key) * scale * factor(rec["tool"], inflight)
            rec["end"] = rec["t"] + rec["dur"]

    def fire(kind, upcoming, step=None):
        if launcher:
            got = launcher({"kind": kind, "state": state, "trace": trace, "step": step, "upcoming": upcoming,
                            "index": len(triggers)})
            triggers.append(len(got))
            launch_all([(tool, args, T + delay) for tool, args, delay in got], "spec")

    for turn in trace["turns"]:
        if not turn["steps"]:
            continue
        start, prev = T, turn["t_start"]
        state.start_turn(turn["user"])
        upcoming = [c for s in turn["steps"] for c in s["calls"]]
        done = 0
        fire("turn", upcoming)
        for step in turn["steps"]:
            T += step["t_start"] - prev
            fire("llm", upcoming[done:], step)
            if use_sptc:
                launch_all([(s["tool"], s["args"], T + s["offset"]) for s in step["sptc"]], "sptc")
            T += step["llm"]["latency_s"]
            prev = step["t_start"] + step["llm"]["latency_s"]
            for c in step["calls"]:
                T += max(0.0, c["t_request"] - prev)
                L = call_duration(trace, c) * scale
                key = (c["tool"], canonical(c["args"]))
                if c["tool"] not in read:
                    table.clear()
                rec = table.get(key)
                hit = rec is not None and rec["t"] <= T
                wait = max(0.0, rec["end"] - T) if hit else L
                if hit:
                    rec["used"] = True
                    del table[key]
                calls.append({"tool": c["tool"], "args": c["args"], "read": c["tool"] in read, "hit": hit, "wait": wait,
                              "latency": L, "source": rec["source"] if hit else None,
                              "seen": seen(state.tokens, c["tool"]) if seen else None})
                T += wait
                prev = c["t_done"]
                state.add_call(c["tool"], parse_json(c["content"]))
                done += 1
                fire("call", upcoming[done:])
            T += max(0.0, step["t_end"] - prev)
            prev = step["t_end"]
        turn_times.append(round(T - start, 3))
    return {"turns": turn_times, "calls": calls, "launches": launches, "triggers": triggers}


def predictor_launcher(pred, tau=None, guard=False):
    """Counting-table guesses at turn start and after each call; guard keeps only IDs seen this turn."""
    def launcher(ev):
        if ev["kind"] == "llm":
            return []
        guesses = pred.predict(ev["state"], tau=tau)
        if guard:
            guesses = [g for g in guesses if all(str(v) in ev["state"].turn_text for v in g[1].values())]
        return [(tool, args, 0.0) for tool, args, _ in guesses]
    return launcher


def candidates(state, specs, read):
    """READ calls whose single ID argument (or no argument) is already known from this episode's text."""
    pool = []
    for tool in sorted(read):
        params = specs[tool]
        if not params:
            pool.append((tool, {}))
        elif len(params) == 1 and params[0] in ID_PATTERNS:
            pool += [(tool, {params[0]: v}) for v in sorted(set(re.findall(ID_PATTERNS[params[0]], state.text)))]
    return pool


def random_launcher(counts, specs, read, seed):
    """Placebo: as many early calls per trigger as the predictor made, picked at random from known IDs."""
    rng = random.Random(seed)

    def launcher(ev):
        n = counts[ev["index"]] if ev["index"] < len(counts) else 0
        pool = candidates(ev["state"], specs, read) if n else []
        return [(tool, args, 0.0) for tool, args in rng.sample(pool, min(n, len(pool)))]
    return launcher


def budget_at(counts, ev):
    return counts[ev["index"]] if ev["index"] < len(counts) else 0


def control_args(rng, pred, tool, params, known):
    """The args this tool is usually called with: usual literals as constants, the rest random known values (None if none)."""
    args = {}
    for a in pred.arg_names.get(tool, params):
        const = pred.const_for(tool, a)
        if const is not None:
            args[a] = const
        elif known:
            args[a] = rng.choice(known)
        else:
            return None
    return args


def random_tool_launcher(pred, counts, seed):
    """Control 1: a random safe tool per budgeted slot, with control_args."""
    rng = random.Random(seed)

    def launcher(ev):
        tools, specs, got = sorted(ev["trace"]["read_tools"]), ev["trace"]["tool_specs"], []
        for _ in range(budget_at(counts, ev) if tools else 0):
            tool = rng.choice(tools)
            args = control_args(rng, pred, tool, specs[tool], ev["state"].known)
            got += [(tool, args, 0.0)] if args is not None else []
        return got
    return launcher


def frequency_launcher(pred, counts, seed):
    """Control 2: the next tool by plain transition frequency among safe tools, random known values as arguments."""
    rng = random.Random(seed)

    def launcher(ev):
        probs = {t: p for t, p in pred.next_probs(ev["state"].tokens).items() if t in ev["trace"]["read_tools"]}
        got = []
        for _ in range(budget_at(counts, ev) if probs else 0):
            tool = rng.choices(list(probs), weights=list(probs.values()))[0]
            args = control_args(rng, pred, tool, ev["trace"]["tool_specs"][tool], ev["state"].known)
            got += [(tool, args, 0.0)] if args is not None else []
        return got
    return launcher


def wrong_arg_launcher(pred, counts, seed, tau=None):
    """Control 3: the predictor's tools, but every non-literal argument is swapped for a different known value."""
    rng = random.Random(seed)

    def launcher(ev):
        got = []
        for tool, args, _ in pred.predict(ev["state"], tau=tau)[: budget_at(counts, ev)]:
            free = [a for a in args if pred.const_for(tool, a) is None]
            others = {a: [v for v in ev["state"].known if v != str(args[a])] for a in free}
            if free and all(others.values()):
                got.append((tool, {**args, **{a: rng.choice(others[a]) for a in free}}, 0.0))
        return got
    return launcher


def exhaustive_launcher(k, pred):
    """Control 5: every safe tool whose usual args have at most one non-literal, filled with the most recent known values,
    at most k per trigger (brute force, no prediction)."""
    def launcher(ev):
        if ev["kind"] == "llm":
            return []
        specs, pool = ev["trace"]["tool_specs"], []
        for t in sorted(ev["trace"]["read_tools"]):
            names = pred.arg_names.get(t, specs[t])
            consts = {a: pred.const_for(t, a) for a in names}
            free = [a for a in names if consts[a] is None]
            fixed = {a: c for a, c in consts.items() if c is not None}
            if not free:
                pool.append((t, fixed, 0.0))
            elif len(free) == 1:
                pool += [(t, {**fixed, free[0]: v}, 0.0) for v in reversed(ev["state"].known[-k:])]
        return pool[:k]
    return launcher


def oracle_launcher(k):
    """Ceiling: the next k real READ calls of this turn, known in advance."""
    def launcher(ev):
        if ev["kind"] == "llm":
            return []
        read = set(ev["trace"]["read_tools"])
        return [(c["tool"], c["args"], 0.0) for c in ev["upcoming"] if c["tool"] in read][:k]
    return launcher


def ask_self(llm, trace, step):
    """The agent predicting its own next calls from the exact prefix it saw: ([(tool, args)], latency_s)."""
    msgs = [dict(m) for m in trace["agent_messages"][: step["n_messages"]]]
    msgs[-1]["content"] = (msgs[-1]["content"] or "") + "\n\n" + SELF_PROMPT
    reply = llm.chat(msgs)
    parsed = extract_json(reply["content"]) or {}
    calls = [(c["tool"], c.get("args") or {}) for c in parsed.get("calls", []) if isinstance(c, dict) and "tool" in c]
    return calls[:3], reply["latency_s"]


def self_launcher(llm):
    def launcher(ev):
        if ev["kind"] != "llm":
            return []
        calls, latency = ask_self(llm, ev["trace"], ev["step"])
        return [(tool, args, latency) for tool, args in calls]
    return launcher


def hit_at_k(trace, guess):
    """Before each agent LLM step that writes code: is the script's first call in the top 1 / top 3 guesses?"""
    rows, state = [], State()
    for turn in trace["turns"]:
        state.start_turn(turn["user"])
        for step in turn["steps"]:
            if step["calls"]:
                first = (step["calls"][0]["tool"], canonical(step["calls"][0]["args"]))
                ranked = [(tool, canonical(args)) for tool, args in guess(state, step)]
                rows.append([first in ranked[:1], first in ranked[:3]])
            for c in step["calls"]:
                state.add_call(c["tool"], parse_json(c["content"]))
    return rows


def run_replay(cfg, make_predictor, self_llm=None):
    """Replay arms on baseline (A) traces over 3 task orders; the predictor learns online from A traces.
    Only A traces are re-timed: in B/C the REPL waits for the sPTC shadow, so their recorded gaps already hold speculation."""
    runs = Path(cfg["paths"]["runs"])
    traces = {arm: {p.stem: json.loads(p.read_text()) for p in (runs / arm).glob("*.json")} for arm in "ABC"}
    ids = [t for t in json.loads((runs / "order.json").read_text()) if all(t in traces[a] for a in "ABC")]
    out = {"orders": [], "hitk": {"pred": [], "self": []}, "tau": {}}
    for o in range(cfg["replay"]["orders"]):
        order = ids if o == 0 else random.Random(o).sample(ids, len(ids))
        pred, rows = make_predictor(), {}
        for tid in order:
            A = traces["A"][tid]
            r = {"A": simulate(A), "pred_alone": simulate(A, predictor_launcher(pred), seen=pred.seen),
                 "oracle": simulate(A, oracle_launcher(pred.k)),
                 "guard": simulate(A, predictor_launcher(pred, guard=True), seen=pred.seen)}
            r["random_k"] = simulate(A, random_launcher(r["pred_alone"]["triggers"], A["tool_specs"],
                                                        set(A["read_tools"]), f"{o}:{tid}"))
            if o == 0:
                out["hitk"]["pred"] += hit_at_k(A, lambda s, step: [(t, a) for t, a, _ in pred.predict(s, tau=0.0, k=3)])
                if self_llm:
                    r["self"] = simulate(A, self_launcher(self_llm))
                    out["hitk"]["self"] += hit_at_k(A, lambda s, step: ask_self(self_llm, A, step)[0])
                for tau in cfg["replay"]["tau_grid"]:
                    out["tau"].setdefault(str(tau), {})[tid] = simulate(A, predictor_launcher(pred, tau=tau))
            rows[tid] = r
            pred.learn(from_trace(A))
        out["orders"].append(rows)
    write_atomic(Path(cfg["paths"]["results"]) / "replay.json", json.dumps(out))
    return out


def replay_command(cfg, use_self=True):
    runs = Path(cfg["paths"]["runs"])
    read = set(json.loads(next((runs / "A").glob("*.json")).read_text())["read_tools"])
    tau_value = json.loads((runs / "tau.json").read_text())["tau"] if (runs / "tau.json").exists() else None
    self_llm = None
    if use_self:
        budget = Budget(runs / "replay_spend.json", cfg["budget"]["replay_usd"])
        self_llm = make_llm(cfg["agent"], cfg["paths"]["cache"], on_spend=budget.spend)
    try:
        run_replay(cfg, lambda: warm_predictor(cfg, read, tau_value), self_llm)
        print(f"wrote {cfg['paths']['results']}/replay.json")
    except BudgetExceeded as e:
        print(f"stopping ({e}); rerun to resume, cached self-predictions are free")


def agent_time(sim):
    return sum(sim["turns"])


def wasted(sim):
    """Tool-seconds spent on speculative calls nobody used."""
    return sum(launch["dur"] for launch in sim["launches"] if not launch["used"])


def load_traces(cfg, ids):
    folder = Path(cfg["paths"]["runs"]) / "baseline" / "A"
    return [json.loads((folder / f"{i}.json").read_text()) for i in ids if (folder / f"{i}.json").exists()]


def new_predictor(cfg, read, tau):
    p = cfg["predictor"]
    return Predictor(read, k=p["k"], tau=tau, min_obs=p["min_obs"], min_precision=p["min_precision"],
                     min_avail=p["min_avail"])


def train_predictor(cfg, n=None, tau=None):
    """Predictor learned from the first n train baselines; tau (if None) by leave-one-task-out on those traces."""
    everything = load_traces(cfg, cfg["split"]["train"])
    read = set().union(*(t["read_tools"] for t in everything))  # each trace lists only its own task's servers
    traces, grid = everything[:n], cfg["replay"]["tau_grid"]
    if tau is None:
        scores = {t: [] for t in grid}
        for j, held in enumerate(traces):
            pred = new_predictor(cfg, read, 0.0)
            for i, other in enumerate(traces):
                if i != j:
                    pred.learn(from_trace(other))
            base = agent_time(simulate(held))
            for t in grid:
                sim = simulate(held, predictor_launcher(pred, tau=t))
                scores[t].append(base - agent_time(sim) - cfg["predictor"]["lambda_waste"] * wasted(sim))
        tau = max(grid, key=lambda t: (statistics.mean(scores[t]) if scores[t] else 0.0, t))
    pred = new_predictor(cfg, read, tau)
    for trace in traces:
        pred.learn(from_trace(trace))
    return pred


def run_mcp_replay(cfg, self_llm=None):
    """Replay the test baseline (A) traces; write and return {results}/replay.json, where sim = simulate()'s output:
    {"test_ids": [...], "tau": chosen tau, "mean_tool_s": float, "mean_llm_s": float,
     "tasks": {task_id: {arm: sim}}  # arms A, pred_frozen, pred_online, random_tool, frequency, wrong_arg,
                                     # exhaustive_<K>, oracle, guard, self (if self_llm), pred_no_slowdown (if calibration.json: all arms are slowed, this one is not)
     "history": {"<n>": {task_id: sim}}, "history_tau": {"<n>": tau},   # pred_frozen per training-set size
     "tau_curve": {"<tau>": {task_id: sim}},                              # reported only, never used to choose
     "delay": {"<scale>": {"A" | "pred_frozen" | "oracle": {task_id: sim}}}}  # all tool durations x scale"""
    r = cfg["replay"]
    traces = load_traces(cfg, cfg["split"]["test"])
    frozen = train_predictor(cfg)
    online = copy.deepcopy(frozen)
    calib = Path(cfg["paths"]["results"]) / "calibration.json"
    slowdown = json.loads(calib.read_text()) if calib.exists() else None

    def sim(trace, launcher=None, **kw):  # every arm uses the calibrated concurrency slowdown when it exists
        return simulate(trace, launcher, slowdown=slowdown, **kw)
    ks = r["exhaustive_k"] if isinstance(r["exhaustive_k"], list) else [r["exhaustive_k"]]
    out = {"test_ids": [t["task_id"] for t in traces], "tau": frozen.tau, "tasks": {}, "history": {}, "history_tau": {},
           "tau_curve": {}, "delay": {}}
    for h in r["history_sizes"]:
        pred = train_predictor(cfg, n=h)
        out["history_tau"][str(h)] = pred.tau
        out["history"][str(h)] = {t["task_id"]: sim(t, predictor_launcher(pred)) for t in traces}
    for t in traces:
        tid = t["task_id"]
        arms = {"A": sim(t), "pred_frozen": sim(t, predictor_launcher(frozen), seen=frozen.seen),
                "pred_online": sim(t, predictor_launcher(online), seen=online.seen)}
        n = arms["pred_frozen"]["triggers"]
        arms["random_tool"] = sim(t, random_tool_launcher(frozen, n, f"{tid}:random_tool"))
        arms["frequency"] = sim(t, frequency_launcher(frozen, n, f"{tid}:frequency"))
        arms["wrong_arg"] = sim(t, wrong_arg_launcher(frozen, n, f"{tid}:wrong_arg"))
        arms.update({f"exhaustive_{k}": sim(t, exhaustive_launcher(k, frozen)) for k in ks})
        arms["oracle"] = sim(t, oracle_launcher(frozen.k))
        arms["guard"] = sim(t, predictor_launcher(frozen, guard=True), seen=frozen.seen)
        if self_llm:
            arms["self"] = sim(t, self_launcher(self_llm))
        if slowdown:  # sensitivity: the same predictor if concurrent calls did not slow servers down
            arms["pred_no_slowdown"] = simulate(t, predictor_launcher(frozen))
        out["tasks"][tid] = arms
        online.learn(from_trace(t))
        for tau in r["tau_grid"]:
            out["tau_curve"].setdefault(str(tau), {})[tid] = sim(t, predictor_launcher(frozen, tau=tau))
        for scale in r["delay_scales"]:
            row = out["delay"].setdefault(str(scale), {"A": {}, "pred_frozen": {}, "oracle": {}})
            row["A"][tid] = sim(t, scale=scale)
            row["pred_frozen"][tid] = sim(t, predictor_launcher(frozen), scale=scale)
            row["oracle"][tid] = sim(t, oracle_launcher(frozen.k), scale=scale)
    tool_s = [call_duration(t, c) for t in traces for turn in t["turns"] for st in turn["steps"] for c in st["calls"]]
    llm_s = [st["llm"]["latency_s"] for t in traces for turn in t["turns"] for st in turn["steps"]]
    out["mean_tool_s"] = statistics.mean(tool_s) if tool_s else 0.0
    out["mean_llm_s"] = statistics.mean(llm_s) if llm_s else 0.0
    write_atomic(Path(cfg["paths"]["results"]) / "replay.json", json.dumps(out))
    return out


def mcp_replay_command(cfg, use_self=True):
    runs = Path(cfg["paths"]["runs"])
    self_llm = None
    if use_self:
        budget = Budget(runs / "replay_spend.json", cfg["budget"]["replay_usd"])
        self_llm = make_llm(cfg["agent"], cfg["paths"]["cache"], on_spend=budget.spend)
    try:
        run_mcp_replay(cfg, self_llm)
        print(f"wrote {cfg['paths']['results']}/replay.json")
    except BudgetExceeded as e:
        print(f"stopping ({e}); rerun to resume, cached self-predictions are free")
