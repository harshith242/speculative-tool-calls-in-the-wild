import pytest

from fastlane.replay import hit_at_k, oracle_launcher, predictor_launcher, simulate

LAT = {"min_s": 1.0, "max_s": 1.0, "sigma": 0.0}


def call(tool, args, t_request, t_done):
    return {"tool": tool, "args": args, "content": "{}", "error": False, "t_request": t_request, "t_done": t_done}


def trace(calls, sptc=()):
    """One turn: a 2 s LLM step whose script makes `calls`, then a 1 s LLM step that replies."""
    end = calls[-1]["t_done"]
    return {"task_id": "t", "read_tools": ["get"], "latency_cfg": LAT, "tool_specs": {"get": ["id"], "put": ["id"]},
            "turns": [{"user": "hi", "t_start": 0.0, "steps": [
                {"t_start": 0.0, "n_messages": 2, "llm": {"latency_s": 2.0}, "sptc": list(sptc), "calls": calls,
                 "t_end": end + 0.1},
                {"t_start": end + 0.1, "n_messages": 4, "llm": {"latency_s": 1.0}, "sptc": [], "calls": [],
                 "t_end": end + 1.1}]}]}


BASE = [call("get", {"id": "1"}, 2.1, 3.1)]


def at_start(tool, args):
    return lambda ev: [(tool, args, 0.0)] if ev["kind"] == "turn" else []


class StubPredictor:
    def __init__(self, guesses):
        self.guesses = guesses

    def predict(self, state, tau=None, k=None):
        return self.guesses


def test_no_speculation_reproduces_the_recorded_turn():
    assert simulate(trace(BASE))["turns"] == [pytest.approx(4.2)]


def test_launch_at_turn_start_hides_the_call_and_shifts_later_events():
    out = simulate(trace(BASE), at_start("get", {"id": "1"}))
    assert out["turns"] == [pytest.approx(3.2)] and out["calls"][0]["hit"]


def test_sptc_launch_mid_stream_hides_part_of_the_wait():
    out = simulate(trace(BASE, sptc=[{"tool": "get", "args": {"id": "1"}, "offset": 1.5}]), use_sptc=True)
    assert out["turns"] == [pytest.approx(3.6)]


def test_write_drops_an_early_launch():
    calls = [call("put", {"id": "1"}, 2.1, 3.1), call("get", {"id": "1"}, 3.2, 4.2)]
    out = simulate(trace(calls), at_start("get", {"id": "1"}))
    assert not out["calls"][1]["hit"] and out["turns"] == [pytest.approx(5.3)]


def test_oracle_launches_the_upcoming_read():
    assert simulate(trace(BASE), oracle_launcher(3))["turns"] == [pytest.approx(3.2)]


def test_ghost_guard_drops_guesses_with_ids_not_seen_this_turn():
    stub = StubPredictor([("get", {"id": "zzz"}, 1.0)])
    assert len(simulate(trace(BASE), predictor_launcher(stub))["launches"]) == 1
    assert simulate(trace(BASE), predictor_launcher(stub, guard=True))["launches"] == []


def test_hit_at_k_checks_the_first_call_of_each_code_step():
    rows = hit_at_k(trace(BASE), lambda state, step: [("put", {"id": "1"}), ("get", {"id": "1"})])
    assert rows == [[False, True]]


def test_earliest_launch_wins_when_a_delayed_guess_is_overtaken_by_sptc():
    late = lambda ev: [("get", {"id": "1"}, 3.0)] if ev["kind"] == "llm" else []
    out = simulate(trace(BASE, sptc=[{"tool": "get", "args": {"id": "1"}, "offset": 1.5}]), late, use_sptc=True)
    assert out["turns"] == [pytest.approx(3.6)] and out["launches"][0]["source"] == "sptc"
