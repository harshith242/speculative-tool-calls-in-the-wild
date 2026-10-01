import time

from fastlane.store import Store


def make(lat):
    calls = []

    def execute(tool, args):
        calls.append(tool)
        return f"{tool}:{args}", False
    return Store(execute, lambda tool, args: lat, {"get"}), calls


def test_hit_on_in_flight_call_waits_only_the_rest():
    store, _ = make(0.3)
    store.speculate("get", {"id": 1}, "pred")
    time.sleep(0.2)
    start = time.monotonic()
    info = store.call("get", {"id": 1})[2]
    assert info["source"] == "pred" and time.monotonic() - start < 0.2


def test_duplicate_speculation_runs_once_and_its_result_is_used_once():
    store, calls = make(0.01)
    store.speculate("get", {"id": 1}, "pred")
    store.speculate("get", {"id": 1}, "sptc")
    assert store.call("get", {"id": 1})[2]["source"] == "pred"
    assert store.call("get", {"id": 1})[2]["source"] == "real"
    assert calls == ["get", "get"] and len(store.log) == 1


def test_write_drops_pending_results():
    store, _ = make(0.01)
    store.speculate("get", {"id": 1}, "pred")
    store.call("cancel", {"id": 1})
    assert store.call("get", {"id": 1})[2]["source"] == "real" and store.log[0]["used"] is False


def test_speculation_never_runs_writes():
    store, calls = make(0.01)
    assert store.speculate("cancel", {"id": 1}, "pred") is None and calls == []


def test_evict_drops_only_the_given_source():
    store, calls = make(0.01)
    store.speculate("get", {"id": 1}, "sptc")
    store.speculate("get", {"id": 2}, "pred")
    store.evict("sptc")
    assert store.call("get", {"id": 1})[2]["source"] == "real" and store.call("get", {"id": 2})[2]["source"] == "pred"


def test_real_tools_skip_the_sleep_and_record_measured_end():
    store = Store(lambda tool, args: (time.sleep(0.05) or "x", False), None, {"get"})
    store.speculate("get", {"id": 1}, "pred")
    info = store.call("get", {"id": 1})[2]
    assert 0.04 < info["duration"] < 0.2 and store.log[0]["t_end"] >= store.log[0]["t_launch"]


def test_predictor_cap_counts_only_capped_sources():
    store = Store(lambda tool, args: ("x", False), None, {"get"}, cap=2)
    for i in range(4):
        store.speculate("get", {"id": i}, "pred")
        store.speculate("get", {"id": 10 + i}, "sptc")
    assert sum(e["source"] == "pred" for e in store.log) == 2 and sum(e["source"] == "sptc" for e in store.log) == 4


def test_speculative_failures_are_classified():
    store = Store(lambda tool, args: ("HTTP 429 Too Many Requests", True), None, {"get"})
    store.speculate("get", {"id": 1}, "pred").result()
    assert store.log[0]["failure"] == "rate_limit"
