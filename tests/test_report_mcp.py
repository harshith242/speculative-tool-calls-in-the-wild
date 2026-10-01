import json

import pytest

from fastlane.report_mcp import live_arm, sim_arm, stats, write_mcp_report


def call(tool, args, source="real", read=True):
    return {"tool": tool, "args": args, "content": json.dumps({"id": "r1"}), "error": False, "read": read, "source": source,
            "t_request": 1.0, "t_launch": 1.0, "t_done": 2.0, "duration": 1.0, "latency": 1.0}


def spec(args, used, t_end=3.0, failure=None):
    return {"tool": "s__get", "args": args, "source": "pred", "t_launch": 1.0, "t_end": t_end, "used": used, "failure": failure}


def trace(tid, arm, server, total, calls, spec_log=()):
    step = {"t_start": 0.0, "llm": {"latency_s": 0.5}, "sptc": [], "t_end": total, "calls": calls}
    return {"task_id": tid, "arm": arm, "servers": [server], "read_tools": [f"{server}__get"], "tool_specs": [], "latency_cfg": None,
            "turns": [{"user": "look up topic", "t_start": 0.0, "t_end": total, "reply": "done", "steps": [step]}],
            "spec_log": list(spec_log), "termination": "answer"}


def sim(total, hit, launches=()):
    c = {"tool": "s__get", "args": {"q": "topic"}, "read": True, "hit": hit, "wait": 0.0, "latency": 1.0,
         "source": "spec" if hit else None, "seen": hit}
    return {"turns": [total], "calls": [c], "launches": list(launches), "triggers": [1]}


def launch(args, used, dur=1.0):
    return {"tool": "s__get", "args": args, "t": 0.0, "source": "spec", "used": used, "dur": dur, "end": dur}


def test_live_amplification_waste_and_entities():
    calls = [call("s__get", {"q": "a"}, source="pred"), call("s__get", {"q": "b"})]
    log = [spec({"q": "a"}, True), spec({"q": "x"}, False, t_end=4.0), spec({"q": "b"}, False, t_end=2.5)]
    m = live_arm(trace("t", "C", "s", 5.0, calls, log))
    assert m["executed"] == 4 and m["calls"] == 2  # 3 launches + 1 real execution over 2 needed calls
    assert m["wasted_s"] == pytest.approx(3.0 + 1.5) and m["abandoned"] == 2
    assert m["entities"] == 1  # "x" never used for real; "b" was
    assert (m["hits"], m["reads"]) == (1, 2)


def test_sim_waste_per_second_saved_and_na_without_savings():
    a = {"t": sim_arm(sim(10.0, False))}
    arm = {"t": sim_arm(sim(6.0, True, [launch({"q": "topic"}, True), launch({"q": "z"}, False, dur=2.0)]))}
    s = stats(arm, a)
    assert s["saved"] == pytest.approx(4.0) and s["per_saved"] == pytest.approx(0.5)
    assert s["amp"] == pytest.approx(2.0)  # 2 launches + 0 real executions over 1 call
    assert stats(a, arm)["per_saved"] is None


def test_wasted_without_hits_is_not_reported_as_efficiency():
    slow = {"t": sim_arm(sim(12.0, False, [launch({"q": "z"}, False)]))}
    assert stats(slow, {"t": sim_arm(sim(10.0, False))})["per_saved"] is None


TRAIN, TEST = ["tr1"], ["t1", "t2", "g1"]
SERVER = {"tr1": "wiki", "t1": "wiki", "t2": "wiki", "g1": "game"}


def make_cfg(tmp_path):
    return {"paths": {"runs": str(tmp_path / "runs"), "results": str(tmp_path / "res")}, "split": {"train": TRAIN, "test": TEST}}


def build_fixture(tmp_path):
    runs, res = tmp_path / "runs", tmp_path / "res"
    for tid in TRAIN + TEST:
        s = SERVER[tid]
        d = runs / "baseline" / "A"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.json").write_text(json.dumps(trace(tid, "A", s, 10.0, [call(f"{s}__get", {"q": "topic"})])))
    for arm, factor in (("A", 1.0), ("B", 0.8), ("C", 0.6), ("P", 0.7)):
        d = runs / "live" / arm
        d.mkdir(parents=True, exist_ok=True)
        for i, tid in enumerate(TEST):
            if arm == "P" and tid == "g1":
                continue
            s, total = SERVER[tid], (10.0 + i) * factor
            log = [spec({"q": "x"}, False, failure="timeout")] if arm in "CP" else []
            src = "pred" if arm in "CP" else "real"
            (d / f"{tid}.json").write_text(json.dumps(trace(tid, arm, s, total, [call(f"{s}__get", {"q": "topic"}, source=src)], log)))
    for name, v in (("baseline/spend.json", 0.2), ("live/spend.json", 0.4), ("replay_spend.json", 0.03)):
        (runs / name).write_text(json.dumps({"usd": v}))
    arms = {"A": 1.0, "pred_frozen": 0.6, "pred_online": 0.55, "random_tool": 0.9, "frequency": 0.8, "wrong_arg": 0.85,
            "exhaustive_2": 0.75, "exhaustive_4": 0.7, "oracle": 0.4, "guard": 0.65, "self": 0.7, "pred_no_slowdown": 0.5}

    def sims(scale_of, tau_launch=True):
        return {t: sim(10.0 * scale_of, scale_of < 1, [launch({"q": "z"}, False)] if tau_launch and scale_of < 1 else []) for t in TEST}
    rep = {"test_ids": TEST, "tau": 0.2, "mean_tool_s": 2.0, "mean_llm_s": 1.0,
           "tasks": {t: {a: sim(10.0 * f, f < 1, [launch({"q": "z"}, False)] if f < 1 else []) for a, f in arms.items()} for t in TEST},
           "history": {n: sims(f) for n, f in (("0", 0.9), ("5", 0.7), ("10", 0.6))}, "history_tau": {"0": 0.4, "5": 0.3, "10": 0.2},
           "tau_curve": {tau: sims(f) for tau, f in (("0.1", 0.55), ("0.4", 0.7))},
           "delay": {s: {"A": sims(1.0), "pred_frozen": sims(0.6), "oracle": sims(0.4)} for s in ("0.5", "1", "2")}}
    res.mkdir()
    (res / "replay.json").write_text(json.dumps(rep))
    (res / "structure.md").write_text("# Structure of baseline traces\n\n| task |\n")
    (res / "calibration.json").write_text(json.dumps({"s__get": {"1": 1.0, "2": 1.2}}))
    return res


def test_write_mcp_report_end_to_end(tmp_path):
    res = build_fixture(tmp_path)
    write_mcp_report(make_cfg(tmp_path))
    text = (res / "summary.md").read_text()
    for title in ("Primary endpoints (live)", "Live arms", "Speculative failures by kind", "Replay primary", "Ladder (replay)",
                  "Seen vs unseen server", "Hit rate by argument source", "Learning curves", "τ curve", "Delay ratio",
                  "Replay check", "Per-class results", "Falsification checklist", "## Structure", "Calibration"):
        assert title in text
    assert "3 test tasks" in text and "baseline $0.200" in text and "timeout 3" in text
    assert "| exhaustive_2 |" in text and "| unseen (a server not in train) | 1 |" in text
    assert "Skipped" not in text
    for png in ("latency_cdf.png", "history_curve.png", "delay_ratio.png"):
        assert (res / png).stat().st_size > 0


def test_report_skips_sections_without_replay(tmp_path):
    res = build_fixture(tmp_path)
    (res / "replay.json").unlink()
    write_mcp_report(make_cfg(tmp_path))
    text = (res / "summary.md").read_text()
    assert "## Ladder (replay)\n\n_Skipped" in text and "Primary endpoints (live)" in text
