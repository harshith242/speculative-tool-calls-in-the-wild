"""v2 report: live A/B/C/P traces and replay.json -> results_mcp/summary.md plus latency_cdf, history_curve, delay_ratio PNGs.
Primary endpoints are geometric-mean agent-time ratios with task-clustered CIs; every other section is descriptive."""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fastlane.common import canonical  # noqa: E402
from fastlane.files import write_atomic  # noqa: E402
from fastlane.report import P50, leaks, log_speedup  # noqa: E402
from fastlane.structure import SOURCE_ORDER, arg_sources, flat, shape  # noqa: E402

LABELS = {"A": "baseline", "B": "sPTC", "C": "sPTC + predictor", "P": "predictor only"}
LADDER = ["A", "random_tool", "frequency", "wrong_arg", "pred_frozen"]
LADDER_END = ["oracle", "pred_online", "guard", "self", "pred_no_slowdown"]


def P90(x):
    return float(np.percentile(x, 90))


def live_arm(trace):
    """Per-episode metrics of a live trace; executed = real calls + every speculative launch."""
    calls = [c for t in trace["turns"] for s in t["steps"] for c in s["calls"]]
    reads, log = [c for c in calls if c["read"]], trace["spec_log"]
    unused = [e for e in log if not e["used"]]
    turns = [t["t_end"] - t["t_start"] for t in trace["turns"]]
    return {"turns": turns, "agent_time": sum(turns), "hits": sum(c["source"] != "real" for c in reads), "reads": len(reads),
            "calls": len(calls), "launches": len(log), "executed": len(log) + sum(c["source"] == "real" for c in calls),
            "wasted_s": sum((e["t_end"] if e.get("t_end") is not None else e["t_launch"]) - e["t_launch"] for e in unused),
            "abandoned": len(unused), "entities": leaks(log, calls)[1], "seen": 0, "seen_hits": 0,
            "failures": Counter(e["failure"] for e in log if e.get("failure"))}


def sim_arm(sim):
    """Same metrics for a replay sim."""
    reads, unused = [c for c in sim["calls"] if c["read"]], [x for x in sim["launches"] if not x["used"]]
    return {"turns": sim["turns"], "agent_time": sum(sim["turns"]), "hits": sum(c["hit"] for c in reads), "reads": len(reads),
            "calls": len(sim["calls"]), "launches": len(sim["launches"]),
            "executed": len(sim["launches"]) + sum(not c["hit"] for c in sim["calls"]),
            "wasted_s": sum(x["dur"] for x in unused), "abandoned": len(unused),
            "entities": leaks(sim["launches"], sim["calls"])[1],
            "seen": sum(bool(c["seen"]) for c in reads), "seen_hits": sum(c["hit"] for c in reads if c["seen"]),
            "failures": Counter()}


def gm(a, b):
    """Geometric-mean ratio of a/b over shared tasks with positive times, or None."""
    d = [np.log(a[t] / b[t]) for t in a if t in b and a[t] > 0 and b[t] > 0]
    return float(np.exp(np.mean(d))) if d else None


def fmt(x, spec=".3f"):
    return "n/a" if x is None or x != x else format(x, spec)


def times(per):
    return {t: m["agent_time"] for t, m in per.items()}


def stats(per, base):
    """Totals of an arm against the baseline arm over shared tasks: time saved, waste per second saved, amplification."""
    ids = [t for t in per if t in base]
    tot = lambda k: sum(per[t][k] for t in ids)  # noqa: E731
    saved = sum(base[t]["agent_time"] - per[t]["agent_time"] for t in ids)
    return {"saved": saved, "per_saved": tot("wasted_s") / saved if saved > 1e-9 else None, "hits": tot("hits"),
            "reads": tot("reads"), "launches": tot("launches"), "entities": tot("entities"),
            "amp": tot("executed") / tot("calls") if tot("calls") else None, "n": len(ids)}


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def ratio_row(label, a, b):
    """Primary-endpoint row: ratio, 95% CI, sign-flip p."""
    if gm(a, b) is None:
        return [label, "n/a", "n/a", "n/a"]
    r, lo, hi, p = log_speedup(a, b)
    return [label, f"{r:.3f}", f"[{lo:.3f}, {hi:.3f}]", f"{p:.3f}"]


def load_dir(path):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(path).glob("*.json"))}


def spend(path):
    return json.loads(Path(path).read_text())["usd"] if Path(path).exists() else None


def sim_per(rep, arm, ids):
    return {t: sim_arm(rep["tasks"][t][arm]) for t in ids if arm in rep["tasks"][t]}


def curve_per(sims, ids):
    return {t: sim_arm(sims[t]) for t in ids if t in sims}


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def head_section(c):
    s = [spend(c["runs"] / p) for p in ("baseline/spend.json", "live/spend.json", "replay_spend.json")]
    money = ", ".join(f"{n} ${v:.3f}" for n, v in zip(("baseline", "live", "replay"), s) if v is not None) or "not recorded"
    live_ids = sorted(set.intersection(*(set(c["live"][a]) for a in "ABC"))) if all(c["live"].get(a) for a in "ABC") else []
    n_test = len(c["rep"]["test_ids"]) if c["rep"] else len(c["test"])
    tau = c["rep"]["tau"] if c["rep"] else "n/a"
    return [f"{n_test} test tasks ({len(live_ids)} with live A/B/C traces). τ = {tau}. Spend: {money}."]


def primary_live(c):
    live = c["live"]
    if not (live.get("A") and live.get("B")):
        return None
    t = {a: times(live[a]) for a in live}
    rows = [ratio_row(label, t[x], t[y]) for label, x, y in (("B / A (sPTC)", "B", "A"), ("C / B (predictor on top of sPTC)", "C", "B"),
                                                              ("C / A (secondary)", "C", "A")) if x in t and y in t]
    return ["Geometric-mean ratio of agent time per episode over tasks (below 1 is faster), 95% bootstrap CI over tasks, "
            "two-sided sign-flip p.", ""] + table(["Comparison", "Ratio", "95% CI", "p"], rows)


def live_arms(c):
    live = c["live"]
    if not live.get("A"):
        return None
    rows = []
    for a in "ABCP":
        if a in live:
            per, s = live[a], stats(live[a], live["A"])
            turns = np.concatenate([m["turns"] for m in per.values()])
            rows.append([f"{a} ({LABELS[a]})", len(per), f"{np.mean(list(times(per).values())):.1f}", f"{P50(turns):.2f}",
                         f"{P90(turns):.2f}", f"{s['hits']}/{s['reads']}", fmt(s["per_saved"], ".2f"), fmt(s["amp"], ".2f"), s["entities"]])
    head = ["Arm", "Episodes", "Mean agent time s", "p50 turn s", "p90 turn s", "Hits / READ calls", "Wasted tool-s per s saved vs A",
            "Call amplification", "Distinct spec. entities"]
    out = table(head, rows)
    out += ["", "Amplification = (speculative launches + real executions) / calls; wasted tool-s are unused launches, t_end - t_launch.",
            "", "### Speculative failures by kind", ""]
    frows = []
    for a in "BCP":
        if a in live:
            f, n = sum((m["failures"] for m in live[a].values()), Counter()), sum(m["launches"] for m in live[a].values())
            frows.append([a, n, sum(m["abandoned"] for m in live[a].values()), sum(f.values()),
                          ", ".join(f"{k} {v}" for k, v in sorted(f.items())) or "none", fmt(sum(f.values()) / n if n else None, ".3f")])
    return out + table(["Arm", "Launches", "Abandoned (unused)", "Failed", "By kind", "Failure rate"], frows)


def replay_primary(c):
    rep = c["rep"]
    if not rep:
        return None
    ids, per = rep["test_ids"], {}
    for a in ("pred_frozen", "frequency", "random_tool"):
        per[a] = times(sim_per(rep, a, ids))
    rows = [ratio_row(f"pred_frozen / {y}", per["pred_frozen"], per[y]) for y in ("frequency", "random_tool") if per[y]]
    return ["Does prediction quality matter beyond starting calls early? Ratios below 1 favour the predictor.", ""] + \
        table(["Comparison", "Ratio", "95% CI", "p"], rows)


def ladder_names(rep):
    ex = sorted({a for t in rep["tasks"].values() for a in t if a.startswith("exhaustive_")}, key=lambda a: int(a.split("_")[1]))
    return [a for a in LADDER + ex + LADDER_END if all(a in t for t in rep["tasks"].values())]


def ladder(c):
    rep = c["rep"]
    if not rep:
        return None
    ids, rows = rep["test_ids"], []
    base = sim_per(rep, "A", ids)
    for a in ladder_names(rep):
        per = sim_per(rep, a, ids)
        s = stats(per, base)
        rows.append([a, fmt(gm(times(per), times(base))), f"{s['hits']}/{s['reads']}", s["launches"], fmt(s["per_saved"], ".2f"),
                     fmt(s["amp"], ".2f"), s["entities"]])
    c["ladder"] = ladder_names(rep)
    return table(["Arm", "Agent time vs A (geo-mean)", "Hits / READ", "Launches", "Wasted tool-s per s saved", "Amplification",
                  "Distinct spec. entities"], rows)


def seen_unseen(c):
    rep, base = c["rep"], c["base"]
    if not rep or not base:
        return None
    train = {s for t in c["cfg"]["split"]["train"] if t in base for s in shape(base[t])["servers"]}
    pred, a = times(sim_per(rep, "pred_frozen", rep["test_ids"])), times(sim_per(rep, "A", rep["test_ids"]))
    groups = {"seen (all servers in train)": {t for t in pred if t in base and set(shape(base[t])["servers"]) <= train}}
    groups["unseen (a server not in train)"] = set(pred) - groups["seen (all servers in train)"]
    c["seen_groups"] = {k: gm({t: pred[t] for t in v}, a) for k, v in groups.items()}
    return table(["Test tasks", "n", "pred_frozen / A"],
                 [[k, len(v), fmt(c["seen_groups"][k])] for k, v in groups.items()])


def hit_by_source(c):
    rep, base = c["rep"], c["base"]
    if not rep or not base:
        return None
    n, hit = Counter(), Counter()
    for t in rep["test_ids"]:
        calls, srcs = rep["tasks"][t]["pred_frozen"]["calls"], arg_sources(base[t]) if t in base else []
        if len(calls) != len(srcs):
            continue
        for call, s in zip(calls, srcs):
            if call["read"]:
                kind = next((k for k in reversed(SOURCE_ORDER) if k in {v[0] for v in s.values()}), "user")
                n[kind] += 1
                hit[kind] += bool(call["hit"])
    return table(["Argument source", "READ calls", "Hits", "Hit rate"],
                 [[k, n[k], hit[k], fmt(hit[k] / n[k] if n[k] else None, ".2f")] for k in SOURCE_ORDER])


def history(c):
    rep = c["rep"]
    if not rep or not rep.get("history"):
        return None
    ids, a = rep["test_ids"], times(sim_per(rep, "A", rep["test_ids"]))
    rows, xs, ys = [], [], []
    for n in sorted(rep["history"], key=int):
        per = curve_per(rep["history"][n], ids)
        s = stats(per, sim_per(rep, "A", ids))
        rows.append([f"{n} train episodes (frozen)", rep["history_tau"].get(n, "n/a"), fmt(gm(times(per), a)), f"{s['hits']}/{s['reads']}"])
        xs.append(int(n))
        ys.append(gm(times(per), a))
    online = {}
    for label, arm in (("all train, frozen", "pred_frozen"), ("all train + online", "pred_online")):
        per = sim_per(rep, arm, ids)
        if per:
            s = stats(per, sim_per(rep, "A", ids))
            online[arm] = gm(times(per), a)
            rows.append([label, rep["tau"], fmt(online[arm]), f"{s['hits']}/{s['reads']}"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label="frozen, by history size")
    for arm, v in online.items():
        ax.axhline(v, ls="--", color="C1" if arm == "pred_online" else "C2", label=arm)
    ax.set(xlabel="train episodes in history", ylabel="agent time ratio vs A", title="Learning curve (replay)")
    ax.legend()
    save_fig(fig, c["out"] / "history_curve.png")
    return table(["History", "τ chosen", "Agent time vs A", "Hits / READ"], rows)


def tau_curve(c):
    rep = c["rep"]
    if not rep or not rep.get("tau_curve"):
        return None
    ids, base = rep["test_ids"], sim_per(rep, "A", rep["test_ids"])
    rows = []
    for tau in sorted(rep["tau_curve"], key=float):
        per = curve_per(rep["tau_curve"][tau], ids)
        s = stats(per, base)
        rows.append([tau, fmt(gm(times(per), times(base))), fmt(s["hits"] / s["reads"] if s["reads"] else None, ".2f"),
                     fmt(s["per_saved"], ".2f")])
    return ["Measured on test, reported only, never used to choose τ.", ""] + \
        table(["τ", "Agent time vs A", "Hit rate", "Wasted tool-s per s saved"], rows)


def delay_ratio(c):
    rep = c["rep"]
    if not rep or not rep.get("delay") or not rep.get("mean_llm_s"):
        return None
    rows, xs, pred, orc = [], [], [], []
    for scale in sorted(rep["delay"], key=float):
        d = {a: times(curve_per(rep["delay"][scale][a], rep["test_ids"])) for a in ("A", "pred_frozen", "oracle")}
        x = float(scale) * rep["mean_tool_s"] / rep["mean_llm_s"]
        xs.append(x)
        pred.append(gm(d["pred_frozen"], d["A"]))
        orc.append(gm(d["oracle"], d["A"]))
        rows.append([scale, f"{x:.2f}", fmt(pred[-1]), fmt(orc[-1])])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, pred, marker="o", label="pred_frozen / A")
    ax.plot(xs, orc, marker="o", label="oracle / A")
    ax.set(xscale="log", xlabel="tool latency / LLM latency", ylabel="agent time ratio", title="Delay-ratio curve (replay)")
    ax.legend()
    save_fig(fig, c["out"] / "delay_ratio.png")
    return table(["Scale", "Tool / LLM latency", "pred_frozen / A", "oracle / A"], rows)


def replay_check(c):
    rep, live = c["rep"], c["live"]
    if not rep or not live.get("P"):
        return None
    rows, errs = [], []
    for t, m in sorted(live["P"].items()):
        if t in rep["tasks"]:
            r = sum(rep["tasks"][t]["pred_frozen"]["turns"])
            errs.append(abs(r - m["agent_time"]) / max(m["agent_time"], 1e-9))
            rows.append([t, f"{m['agent_time']:.1f}", f"{r:.1f}", f"{errs[-1]:.1%}"])
    if not rows:
        return None
    return ["P is the predictor without sPTC and used the frozen (train-only) predictor, like replay pred_frozen.", ""] + \
        table(["Task", "Live P s", "Replay pred_frozen s", "Abs. error"], rows) + ["", f"Median absolute error: {np.median(errs):.1%}."]


def per_class(c):
    rep, base = c["rep"], c["base"]
    if not rep or not base:
        return None
    classes = {t: shape(base[t])["classes"] for t in rep["test_ids"] if t in base}
    pred, a = times(sim_per(rep, "pred_frozen", rep["test_ids"])), times(sim_per(rep, "A", rep["test_ids"]))
    live = c["live"]
    rows = []
    for label in sorted({x for v in classes.values() for x in v}):
        ids = {t for t, v in classes.items() if label in v}
        cb = gm({t: live["C"][t]["agent_time"] for t in ids if t in live.get("C", {})},
                {t: live["B"][t]["agent_time"] for t in ids if t in live.get("B", {})}) if live.get("C") and live.get("B") else None
        rows.append([label, len(ids), fmt(gm({t: pred[t] for t in ids if t in pred}, a)), fmt(cb)])
    return table(["Class", "Tasks", "Replay pred_frozen / A", "Live C / B"], rows)


def structure_section(c):
    path = c["out"] / "structure.md"
    if not path.exists():
        return None
    return [("##" + line if line.startswith("#") else line) for line in path.read_text().splitlines()]


def calibration_section(c):
    path = c["out"] / "calibration.json"
    if not path.exists():
        return None
    cal = json.loads(path.read_text())
    levels = sorted({int(k) for v in cal.values() for k in v})
    rows = [[tool] + [fmt(v.get(str(lv)), ".2f") for lv in levels] for tool, v in sorted(cal.items())]
    return ["Slowdown factor of a call when k identical calls run at once.", ""] + table(["Tool"] + [f"k={lv}" for lv in levels], rows)


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None


def verdict(flag, evidence):
    return f"**{'n/a' if flag is None else 'yes' if flag else 'no'}** ({evidence})"


def falsification(c):
    rep, live = c["rep"], c["live"]
    if not rep:
        return None
    ids = rep["test_ids"]
    base = sim_per(rep, "A", ids)
    pf = sim_per(rep, "pred_frozen", ids)
    lines = ["Each line is a condition that would count against a positive C result; yes means it holds.", ""]
    r_pf = gm(times(pf), times(base))
    for arm in ("frequency", "wrong_arg"):
        r = gm(times(sim_per(rep, arm, ids)), times(pf))
        lines.append(f"- {arm} within 5% of the predictor (or faster): " + verdict(None if r is None else r <= 1.05, f"{arm} / pred_frozen = {fmt(r)}"))
    cb = gm(times(live.get("C", {})), times(live.get("B", {})))
    lines.append("- live C fails to beat B while replay says it should: " +
                 verdict(None if cb is None or r_pf is None else cb >= 1 and r_pf < 1, f"live C/B = {fmt(cb)}, replay pred_frozen/A = {fmt(r_pf)}"))
    ca = gm(times(live.get("C", {})), times(live.get("A", {})))
    lines.append(f"- live C vs replay disagreement (info): live C/A = {fmt(ca)}, replay pred_frozen/A = {fmt(r_pf)}")
    for label, a, b in (("replay pred_frozen vs A", pf, base), ("live C vs B", live.get("C", {}), live.get("B", {}))):
        saved = sorted((b[t]["agent_time"] - a[t]["agent_time"] for t in a if t in b), reverse=True)
        share = sum(saved[:3]) / sum(saved) if saved and sum(saved) > 1e-9 else None
        lines.append(f"- gain concentrated in 2-3 tasks, {label} (top-3 share above 50%): " +
                     verdict(None if share is None else share > 0.5, f"top-3 share of total saved time = {fmt(share, '.0%')} over {len(saved)} tasks"))
    rate = {}
    for a in "BC":
        if live.get(a):
            f = sum(sum(m["failures"].values()) for m in live[a].values())
            n = sum(m["launches"] for m in live[a].values())
            rate[a] = (f / n if n else 0.0, f, n)
    if len(rate) == 2:
        lines.append("- C raises the speculative failure rate vs B: " + verdict(
            rate["C"][0] > rate["B"][0], f"C {rate['C'][1]}/{rate['C'][2]} vs B {rate['B'][1]}/{rate['B'][2]}"))
    seq = lambda t: [(x["tool"], canonical(x["args"])) for _, _, x in flat(t)]  # noqa: E731
    if c["live_traces"].get("A"):
        same = {a: sum(seq(tr) == seq(c["live_traces"]["A"][t]) for t, tr in c["live_traces"].get(a, {}).items() if t in c["live_traces"]["A"])
                for a in "BC" if c["live_traces"].get(a)}
        lines.append("- C trajectories differ from A/B on identical tool outputs (info, LLM sampling can also differ): real-call sequences equal to A on " +
                     ", ".join(f"{a} {n}/{len(c['live_traces'][a])}" for a, n in same.items()))
    arms = [a for a in c.get("ladder", []) if a != "A"]
    st = [stats(sim_per(rep, a, ids), base) for a in arms]
    r_l, r_h = corr([s["saved"] for s in st], [s["launches"] for s in st]), corr([s["saved"] for s in st], [s["hits"] for s in st])
    lines.append("- gain tracks the number of speculative calls rather than hit quality (across ladder arms): " + verdict(
        None if r_l is None or r_h is None else r_l > r_h, f"corr(saved, launches) = {fmt(r_l, '.2f')}, corr(saved, hits) = {fmt(r_h, '.2f')}"))
    sg = c.get("seen_groups", {})
    seen, new = (sg.get(k) for k in ("seen (all servers in train)", "unseen (a server not in train)"))
    sh, s_n = sum(m["seen_hits"] for m in pf.values()), sum(m["seen"] for m in pf.values())
    r_h_all, r_n = sum(m["hits"] for m in pf.values()), sum(m["reads"] for m in pf.values())
    lines.append("- gain appears only on seen servers or transitions: " + verdict(
        None if seen is None or new is None else seen < 1 <= new,
        f"pred_frozen/A seen servers {fmt(seen)}, unseen {fmt(new)}; hits on seen calls {sh}/{s_n}, new calls {r_h_all - sh}/{r_n - s_n}"))
    return lines


SECTIONS = [("Primary endpoints (live)", primary_live), ("Live arms", live_arms), ("Replay primary", replay_primary),
            ("Ladder (replay)", ladder), ("Seen vs unseen server", seen_unseen), ("Hit rate by argument source", hit_by_source),
            ("Learning curves (history)", history), ("τ curve", tau_curve), ("Delay ratio", delay_ratio),
            ("Replay check (live P vs replay)", replay_check), ("Per-class results", per_class),
            ("Falsification checklist", falsification), ("Structure", structure_section), ("Calibration", calibration_section)]


def write_mcp_report(cfg):
    runs, out = Path(cfg["paths"]["runs"]), Path(cfg["paths"]["results"])
    test = cfg["split"]["test"]
    live_ids = test + (cfg["split"]["train"] if cfg.get("live_tasks") == "all" else [])  # sPTC learns nothing, so no split
    traces = {a: {t: tr for t, tr in load_dir(runs / "live" / a).items() if t in live_ids} for a in "ABCP"}
    base = {t: tr for t, tr in load_dir(runs / "baseline" / "A").items() if t in cfg["split"]["train"] + test}
    rep = json.loads((out / "replay.json").read_text()) if (out / "replay.json").exists() else None
    c = {"cfg": cfg, "runs": runs, "out": out, "test": test, "base": base, "rep": rep,
         "live_traces": {a: t for a, t in traces.items() if t},
         "live": {a: {t: live_arm(tr) for t, tr in ts.items()} for a, ts in traces.items() if ts}}
    lines = ["# FastLane v2 results (MCP-Bench, real tools)", ""] + head_section(c)
    for title, fn in SECTIONS:
        body = fn(c)
        lines += ["", f"## {title}", ""] + (body if body is not None else ["_Skipped: inputs missing._"])
    write_atomic(out / "summary.md", "\n".join(lines) + "\n")
    if c["live"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for a in "ABC":
            if a in c["live"]:
                x = np.sort(np.concatenate([m["turns"] for m in c["live"][a].values()]))
                ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=LABELS[a])
        ax.set(xlabel="agent turn latency (s)", ylabel="share of turns", title="Live turn latency")
        ax.legend()
        save_fig(fig, out / "latency_cdf.png")
    print(f"wrote {out}/summary.md")
