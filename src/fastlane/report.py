"""Results: live arms (reality check) and replay arms (precise: paired on identical traces), with task-clustered CIs.
Writes results/summary.md, results/latency_cdf.png and results/tau_sweep.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fastlane.common import call_latency  # noqa: E402
from fastlane.files import write_atomic  # noqa: E402

LABELS = {"A": "baseline", "B": "sPTC", "C": "sPTC + predictor"}


def P50(x):
    return float(np.percentile(x, 50))


def P95(x):
    return float(np.percentile(x, 95))


def leaks(launches, real_calls):
    """(abandoned speculative calls, distinct IDs they touched that the real episode never used)."""
    real = {str(v) for c in real_calls for v in c["args"].values()}
    abandoned = [x for x in launches if not x["used"]]
    return len(abandoned), len({str(v) for x in abandoned for v in x["args"].values()} - real)


def live_metrics(trace):
    lat = lambda tool, args: call_latency(trace["task_id"], tool, args, trace["latency_cfg"])  # noqa: E731
    turns = [t["t_end"] - t["t_start"] for t in trace["turns"] if t["steps"]]
    calls = [c for t in trace["turns"] for s in t["steps"] for c in s["calls"]]
    reads = [c for c in calls if c["read"]]
    abandoned, leaked = leaks(trace["spec_log"], calls)
    return {"turns": turns, "agent_time": sum(turns), "hits": sum(c["source"] != "real" for c in reads),
            "reads": len(reads), "wasted_s": sum(lat(e["tool"], e["args"]) for e in trace["spec_log"] if not e["used"]),
            "real_s": sum(c["latency"] for c in reads), "abandoned": abandoned, "leaked": leaked,
            "reward": trace.get("reward", 0.0)}


def sim_metrics(sim, trace):
    lat = lambda tool, args: call_latency(trace["task_id"], tool, args, trace["latency_cfg"])  # noqa: E731
    reads = [c for c in sim["calls"] if c["read"]]
    abandoned, leaked = leaks(sim["launches"], sim["calls"])
    return {"turns": sim["turns"], "agent_time": sum(sim["turns"]), "hits": sum(c["hit"] for c in reads),
            "reads": len(reads), "seen": sum(bool(c["seen"]) for c in reads),
            "seen_hits": sum(c["hit"] for c in reads if c["seen"]),
            "wasted_s": sum(lat(x["tool"], x["args"]) for x in sim["launches"] if not x["used"]),
            "real_s": sum(c["latency"] for c in reads), "abandoned": abandoned, "leaked": leaked}


def log_speedup(a, b, n=10000, seed=0):
    """Primary endpoint: per-task log(T_a / T_b) -> (geometric-mean ratio, 95% bootstrap CI over tasks, sign-flip p)."""
    d = np.array([np.log(a[t] / b[t]) for t in sorted(set(a) & set(b)) if a[t] > 0 and b[t] > 0])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n)]
    flips = (rng.choice([-1.0, 1.0], (n, len(d))) * d).mean(axis=1)
    p = float(np.mean(np.abs(flips) >= abs(d.mean()) - 1e-12))
    return float(np.exp(d.mean())), float(np.exp(np.percentile(boots, 2.5))), float(np.exp(np.percentile(boots, 97.5))), p


def paired_ci(a, b, stat, n=10000, seed=0):
    """stat(a's turns) - stat(b's turns) with a task-clustered bootstrap 95% CI; a, b: {task: [turn latencies]}."""
    tids = sorted(t for t in set(a) & set(b) if a[t] and b[t])
    rng = np.random.default_rng(seed)

    def diff(sample):
        return stat(np.concatenate([a[t] for t in sample])) - stat(np.concatenate([b[t] for t in sample]))
    boots = [diff(rng.choice(tids, len(tids))) for _ in range(n)]
    return diff(tids), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def total(per_task, key):
    return sum(m[key] for m in per_task.values())


def arm_row(name, per_task, success=None):
    turns = np.concatenate([m["turns"] for m in per_task.values()])
    cells = [name, len(turns), f"{P50(turns):.2f}", f"{P95(turns):.2f}",
             f"{np.mean([m['agent_time'] for m in per_task.values()]):.1f}",
             f"{total(per_task, 'hits')}/{total(per_task, 'reads')}",
             f"{total(per_task, 'wasted_s') / max(total(per_task, 'real_s'), 1e-9):.2f}",
             f"{total(per_task, 'abandoned')} / {total(per_task, 'leaked')}"]
    if success is not None:
        cells.append(f"{success:.2f}")
    return "| " + " | ".join(str(c) for c in cells) + " |"


def diff_row(name, a, b):
    ta, tb = ({t: m["turns"] for t, m in x.items()} for x in (a, b))
    cells = []
    for stat in (P50, P95):
        point, lo, hi = paired_ci(ta, tb, stat)
        cells.append(f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    return f"| {name} | {cells[0]} | {cells[1]} |"


def write_report(cfg):
    runs, out = Path(cfg["paths"]["runs"]), Path(cfg["paths"]["results"])
    traces = {arm: {p.stem: json.loads(p.read_text()) for p in sorted((runs / arm).glob("*.json"))} for arm in "ABC"}
    ids = sorted(set.intersection(*(set(t) for t in traces.values())))
    live = {arm: {t: live_metrics(traces[arm][t]) for t in ids} for arm in "ABC"}
    rep = json.loads((out / "replay.json").read_text())
    order0 = rep["orders"][0]
    names = [n for n in ("A", "pred_alone", "random_k", "oracle", "guard", "self")
             if all(n in order0[t] for t in order0)]
    base = dict.fromkeys(names, "A")
    sim = {n: {t: sim_metrics(order0[t][n], traces[base[n]][t]) for t in order0} for n in names}
    spend = lambda f: json.loads((runs / f).read_text())["usd"] if (runs / f).exists() else 0.0  # noqa: E731
    tau_text = json.loads((runs / "tau.json").read_text())["tau"] if (runs / "tau.json").exists() else "not saved (history incomplete)"
    correct = sum(m["reward"] for arm in live.values() for m in arm.values())
    retried = sum(t.get("retried", False) for arm in traces.values() for tr in arm.values() for t in tr["turns"])
    head = "| Arm | Turns | p50 s | p95 s | Mean agent time / episode s | Hits / READ calls | Wasted / real tool-s | Abandoned / leaked IDs |"
    lines = [
        "# FastLane results", "",
        f"{len(ids)} tau2 retail tasks. τ = {tau_text}. "
        f"Live spend ${spend('spend.json'):.3f}, replay spend ${spend('replay_spend.json'):.3f}. "
        f"Cost per correct answer ${spend('spend.json') / max(correct, 1):.3f} (live spend / correct episodes, all arms; "
        f"speculation does not change LLM spend). {retried} turns needed an API retry.", "",
        "## Live arms (reality check)", "", head + " Success |", "|" + "---|" * 9]
    lines += [arm_row(f"{a} ({LABELS[a]})", live[a], np.mean([m["reward"] for m in live[a].values()])) for a in "ABC"]
    lines += ["", "## Primary endpoint: agent time per episode, live (geometric-mean ratio, 95% CI over tasks, sign-flip p)", "",
              "| Comparison | Ratio | 95% CI | p |", "|---|---|---|---|"]
    for label, x, y in (("B / A", "B", "A"), ("C / B", "C", "B"), ("C / A", "C", "A")):
        r, lo, hi, pv = log_speedup({t: m["agent_time"] for t, m in live[x].items()}, {t: m["agent_time"] for t, m in live[y].items()})
        lines.append(f"| {label} | {r:.3f} | [{lo:.3f}, {hi:.3f}] | {pv:.3f} |")
    lines += ["", "## Live paired differences, secondary (task-clustered 95% CI, seconds; p95 rests on few tasks)", "",
              "| Comparison | p50 diff | p95 diff |", "|---|---|---|",
              diff_row("C − B", live["C"], live["B"]), diff_row("B − A", live["B"], live["A"]),
              diff_row("C − A", live["C"], live["A"])]
    lines += ["", "## Replay arms: prediction strategies on identical baseline conversations (order 0, no sPTC)", "", head, "|" + "---|" * 8]
    lines += [arm_row(n, sim[n]) for n in names]
    pairs = [("pred_alone − A", "pred_alone", "A"), ("random_k − A", "random_k", "A"),
             ("pred_alone − random_k", "pred_alone", "random_k"), ("oracle − A", "oracle", "A"),
             ("guard − pred_alone", "guard", "pred_alone"), ("self − A", "self", "A"),
             ("pred_alone − self", "pred_alone", "self")]
    lines += ["", "## Replay paired differences (task-clustered 95% CI, seconds)", "",
              "| Comparison | p50 diff | p95 diff |", "|---|---|---|"]
    lines += [diff_row(label, sim[x], sim[y]) for label, x, y in pairs if x in sim and y in sim]
    lines += ["", "## Seen-before vs new READ calls (replay, order 0)", "",
              "| Arm | Hit rate, seen | Hit rate, new |", "|---|---|---|"]
    for n in ("pred_alone", "guard"):
        s, sh = total(sim[n], "seen"), total(sim[n], "seen_hits")
        r, h = total(sim[n], "reads"), total(sim[n], "hits")
        lines.append(f"| {n} | {sh}/{s} | {h - sh}/{r - s} |")
    lines += ["", "## Order spread (replay): predictor alone − A, p50 and p95 per task order", ""]
    for o, rows in enumerate(rep["orders"]):
        c = np.concatenate([rows[t]["pred_alone"]["turns"] for t in rows])
        b = np.concatenate([rows[t]["A"]["turns"] for t in rows])
        lines.append(f"- order {o}: p50 {P50(c) - P50(b):+.2f} s, p95 {P95(c) - P95(b):+.2f} s")
    lines += ["", "## Hit@k before each code step (first call of the script, exact match)", "",
              "| Source | Steps | Hit@1 | Hit@3 |", "|---|---|---|---|"]
    for src, rows in rep["hitk"].items():
        if rows:
            arr = np.array(rows, dtype=float)
            lines.append(f"| {src} | {len(rows)} | {arr[:, 0].mean():.2f} | {arr[:, 1].mean():.2f} |")
    taus = sorted(rep["tau"], key=float)
    sweep = {tau: {t: sim_metrics(s, traces["A"][t]) for t, s in rep["tau"][tau].items()} for tau in taus}
    lines += ["", "## τ sweep (predictor alone, replay)", "", "| τ | Hit rate | Wasted / real tool-s | p50 s | p95 s |",
              "|---|---|---|---|---|"]
    for tau in taus:
        m = sweep[tau]
        turns = np.concatenate([x["turns"] for x in m.values()])
        lines.append(f"| {tau} | {total(m, 'hits') / max(total(m, 'reads'), 1):.2f} | "
                     f"{total(m, 'wasted_s') / max(total(m, 'real_s'), 1e-9):.2f} | {P50(turns):.2f} | {P95(turns):.2f} |")
    lines += ["", "## Simulator check: per-task agent time, replay vs live", ""]
    for name, arm in (("A", "A"),):
        errs = [abs(sim[name][t]["agent_time"] - live[arm][t]["agent_time"]) / max(live[arm][t]["agent_time"], 1e-9)
                for t in ids if t in sim[name]]
        lines.append(f"- {name}: median absolute error {np.median(errs):.1%}")
    write_atomic(out / "summary.md", "\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in "ABC":
        x = np.sort(np.concatenate([m["turns"] for m in live[a].values()]))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=LABELS[a])
    ax.set(xlabel="agent turn latency (s)", ylabel="share of turns", title="Live turn latency")
    ax.legend()
    fig.savefig(out / "latency_cdf.png", dpi=150, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([float(t) for t in taus], [total(sweep[t], "hits") / max(total(sweep[t], "reads"), 1) for t in taus], label="hit rate")
    ax.plot([float(t) for t in taus], [total(sweep[t], "wasted_s") / max(total(sweep[t], "real_s"), 1e-9) for t in taus],
            label="wasted / real tool-s")
    ax.set(xlabel="τ", title="Predictor threshold sweep (replay)")
    ax.legend()
    fig.savefig(out / "tau_sweep.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out}/summary.md")
