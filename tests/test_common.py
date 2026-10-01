from fastlane.common import call_latency, tool_median

LAT = {"min_s": 0.5, "max_s": 5.0, "sigma": 0.3}


def test_latency_is_seeded_per_call():
    a = call_latency("1", "get_order_details", {"order_id": "#W1"}, LAT)
    assert a == call_latency("1", "get_order_details", {"order_id": "#W1"}, LAT)
    assert a != call_latency("1", "get_order_details", {"order_id": "#W2"}, LAT)
    assert a != call_latency("2", "get_order_details", {"order_id": "#W1"}, LAT)


def test_tool_medians_differ_and_stay_in_range():
    meds = [tool_median(t, 0.5, 5.0) for t in ("get_user_details", "get_order_details", "find_user_id_by_email")]
    assert all(0.5 <= m <= 5.0 for m in meds) and len(set(meds)) == 3


def test_fixed_latency_when_range_is_a_point():
    assert call_latency("t", "get", {"id": "1"}, {"min_s": 1.0, "max_s": 1.0, "sigma": 0.0}) == 1.0
