import json

import pytest

from fastlane.predictor import Predictor, from_trace
from fastlane.replay import (exhaustive_launcher, frequency_launcher, predictor_launcher, random_tool_launcher,
                             run_mcp_replay, simulate, train_predictor, wrong_arg_launcher)

SPECS = {"s__get": ["title"], "s__find": ["q"], "s__put": ["title"]}


def call(tool, args, t_request, duration, content="{}"):
    return {"tool": tool, "args": args, "content": content, "error": False, "t_request": t_request,
            "t_done": t_request + duration, "duration": duration}


def trace(calls, user='Tell me about "Hubble" and Mars Rover', tid="t"):
    """One turn: a 2 s LLM step whose script makes `calls`, then a 1 s LLM step that replies; measured durations."""
    end = calls[-1]["t_done"]
    return {"task_id": tid, "read_tools": ["s__get", "s__find"], "latency_cfg": None, "tool_specs": SPECS,
            "agent_messages": [{"role": "user", "content": "hi"}] * 4,
            "turns": [{"user": user, "t_start": 0.0, "steps": [
                {"t_start": 0.0, "n_messages": 2, "llm": {"latency_s": 2.0}, "sptc": [], "calls": calls,
                 "t_end": end + 0.1},
                {"t_start": end + 0.1, "n_messages": 4, "llm": {"latency_s": 1.0}, "sptc": [], "calls": [],
                 "t_end": end + 1.1}]}]}


def at_start(*launches):
    return lambda ev: [(t, a, 0.0) for t, a in launches] if ev["kind"] == "turn" else []


class StubPredictor:
    def __init__(self, probs=None, guesses=(), consts=None):
        self.probs, self.guesses, self.consts, self.arg_names = probs or {}, list(guesses), consts or {}, {}

    def const_for(self, tool, arg):
        return self.consts.get((tool, arg))

    def next_probs(self, tokens):
        return self.probs

    def predict(self, state, tau=None, k=None):
        return self.guesses


def known_trace():
    return trace([call("s__find", {"q": "x"}, 2.1, 1.0, '{"title": "Webb"}'), call("s__get", {"title": "Webb"}, 3.2, 1.0)])


def test_measured_duration_is_used_and_a_launch_can_hide_it_fully():
    t = trace([call("s__get", {"title": "a"}, 2.1, 0.7)])
    assert simulate(t)["turns"] == [pytest.approx(3.9)]
    out = simulate(t, at_start(("s__get", {"title": "a"})))
    assert out["calls"][0]["wait"] == 0 and out["turns"] == [pytest.approx(3.2)]


def test_scale_multiplies_every_duration():
    t = trace([call("s__get", {"title": "a"}, 2.1, 1.0)])
    assert simulate(t, scale=2.0)["calls"][0]["latency"] == 2.0


def test_slowdown_stretches_simultaneous_launches_and_rounds_down_to_a_level():
    calls = [call("s__get", {"title": c}, 2.1, 2.0) for c in "ab"]
    both = at_start(("s__get", {"title": "a"}), ("s__get", {"title": "b"}))
    slow = {"s__get": {"1": 1.0, "2": 1.5}}
    out = simulate(trace(calls), both, slowdown=slow)
    assert [x["dur"] for x in out["launches"]] == [3.0, 3.0] and out["calls"][0]["wait"] == pytest.approx(0.9)
    assert [x["dur"] for x in simulate(trace(calls), both)["launches"]] == [2.0, 2.0]
    three = at_start(*[("s__get", {"title": c}) for c in "abc"])
    assert {x["dur"] for x in simulate(trace(calls + [call("s__get", {"title": "c"}, 2.1, 2.0)]), three,
                                       slowdown=slow)["launches"]} == {3.0}


def test_wrong_guess_gets_the_tools_median_duration_else_the_overall_median():
    calls = [call("s__get", {"title": "a"}, 2.1, 1.0), call("s__get", {"title": "b"}, 3.2, 3.0),
             call("s__get", {"title": "c"}, 6.3, 2.0)]
    out = simulate(trace(calls), at_start(("s__get", {"title": "zzz"}), ("s__find", {"q": "zzz"})))
    assert [x["dur"] for x in out["launches"]] == [2.0, 2.0]


def test_known_values_hold_result_leaves_and_task_phrases():
    seen = []
    simulate(known_trace(), lambda ev: seen.append(list(ev["state"].known)) or [])
    assert {"Hubble", "Mars Rover", "Tell"} <= set(seen[0]) and "Webb" not in seen[0]
    assert "Webb" in seen[-1] and "x" not in seen[-1]  # strings under 2 chars are dropped


def test_frequency_launcher_proposes_the_predictor_top_tool_with_a_known_value():
    out = simulate(known_trace(), frequency_launcher(StubPredictor({"s__get": 0.9, "<reply>": 0.1}), [1], 0))
    assert out["launches"][0]["tool"] == "s__get"
    assert out["launches"][0]["args"]["title"] in {"Hubble", "Mars Rover", "Tell"}


def test_wrong_arg_launcher_keeps_the_tool_but_changes_the_value():
    stub = StubPredictor(guesses=[("s__get", {"title": "Hubble"}, 1.0)])
    out = simulate(known_trace(), wrong_arg_launcher(stub, [1], 0))
    assert out["launches"][0]["tool"] == "s__get" and out["launches"][0]["args"]["title"] in {"Mars Rover", "Tell"}


def test_random_tool_draws_only_safe_tools():
    out = simulate(known_trace(), random_tool_launcher(StubPredictor(), [5] * 10, 0))
    assert out["launches"] and {x["tool"] for x in out["launches"]} <= {"s__get", "s__find"}


def test_exhaustive_launches_at_most_k_per_trigger_and_only_single_param_safe_tools():
    out = simulate(known_trace(), exhaustive_launcher(2, StubPredictor()))
    assert max(out["triggers"]) == 2 and {x["tool"] for x in out["launches"]} <= {"s__get", "s__find"}


def test_count_matched_controls_respect_the_budget_and_are_deterministic():
    counts = [1, 0, 2, 0]
    stub = StubPredictor({"s__get": 1.0}, [("s__get", {"title": "Hubble"}, 1.0)] * 3)
    for make in (lambda: random_tool_launcher(stub, counts, 3), lambda: frequency_launcher(stub, counts, 3),
                 lambda: wrong_arg_launcher(stub, counts, 3)):
        a, b = simulate(known_trace(), make()), simulate(known_trace(), make())
        assert a == b and all(n <= c for n, c in zip(a["triggers"], counts + [0] * 9)) and a["triggers"][0] == 1


def chain(tid, topic):
    """search -> get on the page it returned, 1 s each."""
    return trace([call("s__find", {"q": topic}, 2.1, 1.0, json.dumps({"results": [f"{topic} Page"]})),
                  call("s__get", {"title": f"{topic} Page"}, 3.2, 1.0, '{"text": "body"}')], f'Look up "{topic}"', tid)


def cfg_for(tmp_path, train, test):
    folder = tmp_path / "runs" / "baseline" / "A"
    folder.mkdir(parents=True)
    for i, topic in enumerate(train + test):
        (folder / f"{topic}_id.json").write_text(json.dumps(chain(f"{topic}_id", topic)))
    return {"split": {"train": [f"{t}_id" for t in train], "test": [f"{t}_id" for t in test]},
            "predictor": {"k": 3, "min_obs": 2, "min_precision": 0.6, "min_avail": 2, "lambda_waste": 0.5},
            "replay": {"tau_grid": [0.1, 0.5, 0.9], "history_sizes": [0, 2], "delay_scales": [0.5, 2],
                       "exhaustive_k": [2, 4]},
            "paths": {"runs": str(tmp_path / "runs"), "results": str(tmp_path / "results"), "cache": str(tmp_path / "c")}}


def test_train_predictor_picks_tau_from_the_grid_and_n_limits_the_history(tmp_path):
    cfg = cfg_for(tmp_path, ["Alpha", "Beta", "Gamma"], [])
    pred = train_predictor(cfg)
    assert pred.tau in cfg["replay"]["tau_grid"] and pred.next
    assert train_predictor(cfg, n=0).next == {} and train_predictor(cfg, n=1, tau=0.3).tau == 0.3


class FakeLLM:
    def chat(self, msgs):
        return {"content": '{"calls": []}', "latency_s": 0.5}


def test_run_mcp_replay_writes_every_arm_and_curve(tmp_path):
    cfg = cfg_for(tmp_path, ["Alpha", "Beta"], ["Gamma", "Delta"])
    out = run_mcp_replay(cfg, FakeLLM())
    assert json.loads((tmp_path / "results" / "replay.json").read_text()) == out
    assert out["test_ids"] == ["Gamma_id", "Delta_id"]
    arms = {"A", "pred_frozen", "pred_online", "random_tool", "frequency", "wrong_arg", "exhaustive_2", "exhaustive_4",
            "oracle", "guard", "self"}
    assert all(set(out["tasks"][t]) == arms for t in out["test_ids"])
    assert set(out["history"]) == {"0", "2"} and set(out["tau_curve"]) == {"0.1", "0.5", "0.9"}
    assert set(out["delay"]) == {"0.5", "2"} and set(out["delay"]["2"]) == {"A", "pred_frozen", "oracle"}
    assert out["mean_tool_s"] == 1.0 and out["mean_llm_s"] == 1.5
    a, p = out["tasks"]["Gamma_id"]["A"], out["tasks"]["Gamma_id"]["pred_frozen"]
    assert a["turns"] == [pytest.approx(5.3)] and sum(p["turns"]) < sum(a["turns"])  # predictor hides the get


def test_calibration_file_adds_the_slowdown_arm(tmp_path):
    cfg = cfg_for(tmp_path, ["Alpha", "Beta"], ["Gamma"])
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "calibration.json").write_text(json.dumps({"s__get": {"1": 1.0, "2": 1.5}}))
    assert "pred_no_slowdown" in run_mcp_replay(cfg)["tasks"]["Gamma_id"]


def test_controls_fill_usual_literals_and_keep_them_when_swapping():
    stub = StubPredictor(guesses=[("s__get", {"title": "Hubble", "lang": "en"}, 1.0)], consts={("s__get", "lang"): "en"})
    stub.arg_names = {"s__get": ["title", "lang"]}
    out = simulate(known_trace(), wrong_arg_launcher(stub, [1], 0))
    assert out["launches"][0]["args"]["lang"] == "en" and out["launches"][0]["args"]["title"] != "Hubble"
    ex = simulate(known_trace(), exhaustive_launcher(4, stub))
    assert all(x["args"].get("lang") == "en" for x in ex["launches"] if x["tool"] == "s__get")
