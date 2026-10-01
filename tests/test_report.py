import json

import numpy as np
import pytest

from fastlane.report import leaks, live_metrics, log_speedup, paired_ci, write_report

LAT = {"min_s": 1.0, "max_s": 1.0, "sigma": 0.0}


def test_paired_ci_recovers_a_constant_shift():
    a, b = {"t1": [3.0, 4.0], "t2": [5.0]}, {"t1": [2.0, 3.0], "t2": [4.0]}
    point, lo, hi = paired_ci(a, b, lambda x: float(np.percentile(x, 50)), n=200)
    assert (point, lo, hi) == (pytest.approx(1.0), pytest.approx(1.0), pytest.approx(1.0))


def test_leaks_counts_only_ids_the_real_episode_never_used():
    launches = [{"used": False, "args": {"order_id": "#W1"}}, {"used": False, "args": {"order_id": "#W9"}},
                {"used": True, "args": {"order_id": "#W2"}}]
    assert leaks(launches, [{"args": {"order_id": "#W1"}}]) == (2, 1)


def test_live_metrics_skip_the_stop_turn_and_count_hits_and_waste():
    trace = {"task_id": "t", "latency_cfg": LAT,
             "turns": [{"t_start": 0.0, "t_end": 5.0, "steps": [{"calls": [
                 {"read": True, "source": "sptc", "latency": 2.0, "args": {}},
                 {"read": True, "source": "real", "latency": 1.0, "args": {}},
                 {"read": False, "source": "real", "latency": 1.0, "args": {}}]}]},
                       {"t_start": 9.0, "t_end": 9.0, "steps": []}],
             "spec_log": [{"used": True, "tool": "get", "args": {}}, {"used": False, "tool": "get", "args": {"id": "x"}}]}
    m = live_metrics(trace)
    assert m["turns"] == [5.0] and (m["hits"], m["reads"]) == (1, 2)
    assert (m["wasted_s"], m["real_s"]) == (1.0, 3.0) and (m["abandoned"], m["leaked"]) == (1, 1)


def _trace(tid):
    call = {"read": True, "source": "sptc", "latency": 2.0, "args": {"id": tid}}
    turn = {"t_start": 0.0, "t_end": 4.0, "steps": [{"calls": [call]}]}
    return {"task_id": tid, "latency_cfg": LAT, "turns": [turn], "reward": 1.0,
            "spec_log": [{"used": True, "tool": "get", "args": {"id": tid}}]}


def _sim():
    call = {"read": True, "hit": True, "seen": False, "latency": 2.0, "args": {"id": "x"}}
    return {"turns": [3.0, 4.0], "calls": [call], "launches": [{"used": False, "tool": "get", "args": {"id": "y"}}],
            "triggers": 1}


def test_write_report_runs_end_to_end_on_a_tiny_fixture(tmp_path):
    runs, out, tasks = tmp_path / "runs", tmp_path / "results", ["t1", "t2"]
    for arm in "ABC":
        (runs / arm).mkdir(parents=True)
        for t in tasks:
            (runs / arm / f"{t}.json").write_text(json.dumps(_trace(t)))
    (runs / "tau.json").write_text(json.dumps({"tau": 0.1}))
    (runs / "spend.json").write_text(json.dumps({"usd": 0.5}))
    arms = ["A", "pred_alone", "B", "C", "random_k", "oracle", "guard"]
    per_task = {t: {a: _sim() for a in arms} for t in tasks}
    out.mkdir()
    (out / "replay.json").write_text(json.dumps({
        "orders": [per_task], "hitk": {"pred": [[True, True], [False, True]], "self": []},
        "tau": {"0.1": {t: _sim() for t in tasks}}}))
    write_report({"paths": {"runs": str(runs), "results": str(out)}})
    assert "## τ sweep" in (out / "summary.md").read_text()
    assert (out / "latency_cdf.png").stat().st_size > 0 and (out / "tau_sweep.png").stat().st_size > 0


def test_log_speedup_gives_geometric_ratio_and_small_p_for_a_consistent_gain():
    a = {f"t{i}": 5.0 + i for i in range(8)}
    b = {t: 2 * v for t, v in a.items()}
    ratio, lo, hi, p = log_speedup(a, b, n=2000)
    assert (ratio, lo, hi) == (pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.5)) and p < 0.02
