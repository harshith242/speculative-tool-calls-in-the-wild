import json
import threading
import time

import pytest

from fastlane.structure import arg_sources, calibrate, overlap, shape, write_structure

SEARCH = [{"title": "Hubble"}, {"title": "Webb"}]


def call(tool, args, req, launch, done, content=""):
    return {"tool": tool, "args": args, "content": content, "error": False, "read": True, "source": "real",
            "t_request": req, "t_launch": launch, "t_done": done, "duration": round(done - launch, 3)}


def make_trace(extra_call=None, task_id="t1"):
    step2 = [call("s__get", {"title": "Hubble"}, 4.3, 4.3, 5.3), call("s__get", {"title": "Webb"}, 5.3, 5.3, 6.3)]
    if extra_call:
        step2.append(extra_call)
    return {"task_id": task_id, "arm": "A", "servers": ["s"], "latency_cfg": None, "spec_log": [], "turns": [{
        "user": "Tell me about the telescope", "t_start": 0.0, "t_end": 6.3, "reply": "done", "steps": [
            {"t_start": 0.0, "llm": {"latency_s": 2.0}, "calls": [
                call("s__search", {"q": "telescope"}, 2.1, 2.1, 3.1, json.dumps(SEARCH))]},
            {"t_start": 3.2, "llm": {"latency_s": 1.0}, "calls": step2}]}]}


def test_sources_overlap_and_ceiling_on_the_hand_trace():
    trace = make_trace()
    assert arg_sources(trace) == [{"q": ("user", None)}, {"title": ("result", 0)}, {"title": ("result", 0)}]
    result = overlap(trace, {"s__search", "s__get"})
    assert [r["earliest"] for r in result["calls"]] == [0.0, 3.1, 3.1]
    assert [r["overlap"] for r in result["calls"]] == pytest.approx([1.0, 1.0, 1.0])
    assert result["ceiling"] == pytest.approx(3.0 / 6.3)


def test_overlap_counts_only_safe_tools():
    assert overlap(make_trace(), {"s__get"})["ceiling"] == pytest.approx(2.0 / 6.3)


def test_shape_of_the_hand_trace():
    s = shape(make_trace())
    assert (s["fan_out"], s["chain_depth"], s["servers"], s["calls"]) == (2, 2, ["s"], 3)
    assert s["classes"] == ["F1", "D1"]


def test_model_made_argument_waits_for_the_end_of_the_stream():
    trace = make_trace(call("s__get", {"title": "Hubble telescope history"}, 6.4, 6.4, 7.0))
    assert arg_sources(trace)[3] == {"title": ("model", None)}
    assert overlap(trace, {"s__get"})["calls"][3]["earliest"] == pytest.approx(4.2)


def test_sequential_dependency_inside_one_step_is_not_fan_out():
    trace = make_trace()
    trace["turns"][0]["steps"] = [{"t_start": 0.0, "llm": {"latency_s": 2.0}, "calls": [
        call("s__search", {"q": "telescope"}, 2.1, 2.1, 3.1, json.dumps(SEARCH)),
        call("s__get", {"title": "Webb"}, 3.1, 3.1, 4.1)]}]
    assert shape(trace)["fan_out"] == 1


def test_cross_server_and_deep_chain_labels():
    trace = make_trace()
    trace["turns"][0]["steps"][1]["calls"][1]["tool"] = "other__get"
    trace["turns"][0]["steps"][1]["calls"][1]["content"] = json.dumps({"id": "x9"})
    trace["turns"][0]["steps"][1]["calls"].append(call("other__open", {"id": "x9"}, 6.4, 6.4, 7.0))
    s = shape(trace)
    assert s["chain_depth"] == 3 and s["classes"] == ["F1", "D2", "X"] and s["servers"] == ["other", "s"]


def test_calibrate_factor_grows_with_concurrency():
    lock, inflight = threading.Lock(), [0]

    def execute(tool, args):
        with lock:
            inflight[0] += 1
        time.sleep(0.01)  # let the burst join before reading the load
        n = inflight[0]
        time.sleep(0.02 * n)
        with lock:
            inflight[0] -= 1
        return "ok", False
    out = calibrate(execute, [("s__get", {"t": "a"}), ("s__get", {"t": "a"}), ("s__get", {"t": "b"})], levels=(1, 4))
    assert out["s__get"]["1"] == 1.0
    assert out["s__get"]["4"] > 1.5


def test_write_structure_smoke(tmp_path):
    runs = tmp_path / "runs" / "baseline" / "A"
    runs.mkdir(parents=True)
    (runs / "tr.json").write_text(json.dumps(make_trace(task_id="tr")))
    (runs / "te.json").write_text(json.dumps(make_trace(task_id="te")))
    cfg = {"paths": {"runs": str(tmp_path / "runs")}, "split": {"train": ["tr", "gone"], "test": ["te"]},
           "safe": {"S": ["search", "get"]}}
    out = tmp_path / "out" / "structure.md"
    write_structure(cfg, out)
    text = out.read_text()
    assert "| tr | train | s | 3 | 2 | 2 | F1 D1 | 0.476 |" in text
    assert "| te | test |" in text and "gone" not in text
    assert "## train (n=1)" in text and "median 0.476" in text and "fan-out >= 2: 100%" in text
