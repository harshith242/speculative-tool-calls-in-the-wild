"""Baseline-trace structure analysis: where each call's arguments come from, the overlap ceiling,
fan-out and chain depth, plus the concurrency calibration of tool latency."""
import json
import re
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fastlane.common import canonical, parse_json
from fastlane.files import write_atomic
from fastlane.predictor import walk

SOURCE_ORDER = ("user", "result", "model")


def flat(trace):
    """(turn, step, call) for every call, in order; a call's index is its position here."""
    return [(turn, step, call) for turn in trace["turns"] for step in turn["steps"] for call in step.get("calls", [])]


def slug(name):
    return re.sub(r"\W+", "_", name.lower()).strip("_")


def arg_sources(trace):
    """Per call, {arg: (source, producer index)}: user text, the earliest earlier result holding the value, else model."""
    out, first_leaf = [], {}  # leaf string -> earliest call whose result holds it; user matches need 3+ chars
    for i, (turn, _, call) in enumerate(flat(trace)):
        found = {}
        for name, value in call["args"].items():
            leaves = [v for _, v in walk(value)] if isinstance(value, (dict, list)) else [str(value)]
            where = [("user", None) if len(leaf) >= 3 and leaf in turn["user"] else ("result", first_leaf[leaf]) if leaf in first_leaf
                     else ("model", None) for leaf in leaves]
            kinds = {w[0] for w in where}
            if not where or "model" in kinds:
                found[name] = ("model", None)
            elif "result" in kinds:  # compound values wait for their latest producer
                found[name] = ("result", max(p for k, p in where if k == "result"))
            else:
                found[name] = ("user", None)
        out.append(found)
        for _, leaf in walk(parse_json(call["content"])):
            first_leaf.setdefault(leaf, i)
    return out


def overlap(trace, safe_tools):
    """Per-call earliest safe start and overlap O_i, plus the ceiling sum(O) / episode time."""
    calls, sources, rows = flat(trace), arg_sources(trace), []
    for (turn, step, call), srcs in zip(calls, sources):
        times = [turn["t_start"]]
        for kind, producer in srcs.values():
            if kind == "result":
                times.append(calls[producer][2]["t_done"])
            elif kind == "model":
                times.append(step["t_start"] + step["llm"]["latency_s"])
        earliest = max(times)
        o = max(0.0, min(call["duration"], call["t_request"] - earliest)) if call["tool"] in safe_tools else 0.0
        rows.append({"tool": call["tool"], "earliest": earliest, "overlap": o})
    total = sum(t["t_end"] - t["t_start"] for t in trace["turns"])
    return {"calls": rows, "ceiling": sum(r["overlap"] for r in rows) / total if total else 0.0}


def shape(trace):
    """Fan-out, chain depth (a call that needs a result is 1 + its producer's depth), servers and class labels."""
    calls, sources = flat(trace), arg_sources(trace)
    producers = [[p for k, p in s.values() if k == "result"] for s in sources]
    depth = []
    for ps in producers:
        depth.append(1 + max((depth[p] for p in ps), default=0))
    fan_out, i = 0, 0
    for turn in trace["turns"]:
        for step in turn["steps"]:
            n = len(step.get("calls", []))
            independent = [not any(p >= i for p in producers[i + j]) for j in range(n)]  # no producer in this step
            fan_out = max(fan_out, sum(independent))
            i += n
    chain = max(depth, default=0)
    servers = sorted({c["tool"].split("__")[0] for _, _, c in calls})
    classes = [f"F{min(max(fan_out - 1, 0), 2)}", f"D{min(max(chain - 1, 0), 2)}"]
    if len(servers) > 1:
        classes.append("X")
    return {"calls": len(calls), "fan_out": fan_out, "chain_depth": chain, "servers": servers, "classes": classes}


def calibrate(execute, calls, levels=(1, 2, 4, 8)):
    """{tool: {str(level): median wall time at level / median at 1}}; each call runs `level` times at once."""
    ratios = {}
    for tool, args in {canonical([t, a]): (t, a) for t, a in calls}.values():
        medians = {level: statistics.median(_timed_burst(execute, tool, args, level)) for level in levels}
        ratios.setdefault(tool, {str(l): [] for l in levels})
        for level in levels:
            ratios[tool][str(level)].append(medians[level] / medians[levels[0]])
    return {tool: {l: statistics.median(r) for l, r in per.items()} for tool, per in ratios.items()}


def _timed_burst(execute, tool, args, n):
    """Wall time of each of n identical calls released together."""
    barrier = threading.Barrier(n)

    def one(_):
        barrier.wait()
        t0 = time.perf_counter()
        execute(tool, args)
        return time.perf_counter() - t0
    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(one, range(n)))


def _load(cfg, ids):
    runs = Path(cfg["paths"]["runs"]) / "baseline" / "A"
    return [json.loads((runs / f"{t}.json").read_text()) for t in ids if (runs / f"{t}.json").exists()]


def _aggregate(rows):
    """Markdown lines summarising one split's per-episode rows."""
    if not rows:
        return ["No traces.", ""]
    ceilings, depths = [r["ceiling"] for r in rows], [r["shape"]["chain_depth"] for r in rows]
    mix = Counter(src for r in rows for src in r["call_sources"])
    n_calls = sum(mix.values())
    classes = Counter(c for r in rows for c in r["shape"]["classes"])
    return [f"- Ceiling: median {np.median(ceilings):.3f}, mean {np.mean(ceilings):.3f}",
            "- Calls by argument source: " + ", ".join(f"{s} {mix[s] / n_calls:.0%}" for s in SOURCE_ORDER) if n_calls
            else "- Calls by argument source: no calls",
            f"- Episodes with fan-out >= 2: {np.mean([r['shape']['fan_out'] >= 2 for r in rows]):.0%}",
            f"- Episodes with a result-to-argument hop: {np.mean([d >= 2 for d in depths]):.0%}",
            f"- Chain depth: median {np.median(depths):.1f}, p90 {np.percentile(depths, 90):.1f}",
            "- Classes: " + ", ".join(f"{c} {n}" for c, n in sorted(classes.items())), ""]


def write_structure(cfg, out_path):
    """Per-task table and per-split aggregates of the baseline traces, written as markdown."""
    safe = {f"{slug(server)}__{tool}" for server, tools in cfg["safe"].items() for tool in tools}
    by_split = {}
    for split in ("train", "test"):
        rows = []
        for trace in _load(cfg, cfg["split"][split]):
            sh = shape(trace)
            srcs = [{k for k, _ in s.values()} or {"user"} for s in arg_sources(trace)]
            rows.append({"task": trace["task_id"], "shape": sh, "ceiling": overlap(trace, safe)["ceiling"],
                         "call_sources": [next(k for k in reversed(SOURCE_ORDER) if k in s) for s in srcs]})
        by_split[split] = rows
    lines = ["# Structure of baseline traces", "", "| task | split | servers | calls | fan-out | chain depth | classes | ceiling |",
             "|---|---|---|---|---|---|---|---|"]
    for split, rows in by_split.items():
        lines += [f"| {r['task']} | {split} | {','.join(r['shape']['servers'])} | {r['shape']['calls']} | "
                  f"{r['shape']['fan_out']} | {r['shape']['chain_depth']} | {' '.join(r['shape']['classes'])} | "
                  f"{r['ceiling']:.3f} |" for r in rows]
    for split, rows in by_split.items():
        lines += ["", f"## {split} (n={len(rows)})", ""] + _aggregate(rows)
    write_atomic(out_path, "\n".join(lines) + "\n")
