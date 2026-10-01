import json

from fastlane.predictor import Predictor, State, load_history, walk

READ = {"find_user_id_by_email", "get_user_details", "get_order_details"}


def make(read=READ, k=3, tau=0.3):
    return Predictor(read, k=k, tau=tau, min_obs=3, min_precision=0.6, min_avail=3)


def episode(i):
    email, uid, order = f"u{i}@x.com", f"u_v_{1000 + i}", f"#W000000{i}"
    return [{"user": f"hi, my email is {email}", "calls": [
        {"tool": "find_user_id_by_email", "args": {"email": email}, "result": uid},
        {"tool": "get_user_details", "args": {"user_id": uid}, "result": {"user_id": uid, "orders": [order]}},
        {"tool": "get_order_details", "args": {"order_id": order}, "result": {"order_id": order}}]}]


def trained():
    p = make()
    for i in range(4):
        p.learn(episode(i))
    return p


def test_walk_gives_shared_paths_for_lists_and_id_keyed_dicts():
    assert list(walk("x")) == [("", "x")]
    assert (".orders[*]", "#W1") in list(walk({"orders": ["#W1"]}))
    assert (".variants.*key", "1234567890") in list(walk({"variants": {"1234567890": {"price": 1}}}))


def test_turn_start_guess_fills_its_arg_from_the_user_text():
    s = State()
    s.start_turn("hello, my email is new@y.com")
    assert trained().predict(s)[0][:2] == ("find_user_id_by_email", {"email": "new@y.com"})


def test_chain_from_a_result_and_fan_out_over_a_list():
    p, s = trained(), State()
    s.start_turn("email new@y.com")
    s.add_call("find_user_id_by_email", "n_m_9999")
    assert p.predict(s)[0][:2] == ("get_user_details", {"user_id": "n_m_9999"})
    s.add_call("get_user_details", {"user_id": "n_m_9999", "orders": ["#W1111111", "#W2222222"]})
    assert {g[1]["order_id"] for g in p.predict(s) if g[0] == "get_order_details"} == {"#W1111111", "#W2222222"}


def test_back_off_uses_the_two_token_context_when_it_has_enough_data():
    p = make(read={"a", "b", "c", "d"}, k=1, tau=0.0)
    for _ in range(3):
        for first, last in (("a", "c"), ("d", "d")):
            p.learn([{"user": "", "calls": [{"tool": t, "args": {}, "result": None} for t in (first, "b", last)]}])
    for first, want in (("a", "c"), ("d", "d")):
        s = State()
        s.start_turn("")
        s.add_call(first, None)
        s.add_call("b", None)
        assert p.predict(s)[0][0] == want


def test_unreliable_source_is_not_used():
    p = make(read={"get_order_details"}, tau=0.0)
    for i in range(4):
        used = "#W0000001" if i == 0 else f"#W999999{i}"
        p.learn([{"user": "order #W0000001 please", "calls": [
            {"tool": "get_order_details", "args": {"order_id": used}, "result": {}}]}])
    s = State()
    s.start_turn("order #W0000001 please")
    assert p.predict(s) == []


def test_tau_gate_blocks_low_scores():
    s = State()
    s.start_turn("email new@y.com")
    assert trained().predict(s, tau=1.01) == []


def test_load_history_reads_traces_in_sampled_order(tmp_path):
    folder = tmp_path / "history"
    folder.mkdir()
    for tid, tool in (("7", "get_user_details"), ("3", "get_order_details")):
        trace = {"turns": [{"user": "hi", "steps": [{"calls": [{"tool": tool, "args": {}, "content": "{\"a\": 1}"}]}]}]}
        (folder / f"{tid}.json").write_text(json.dumps(trace))
    (folder / "order.json").write_text(json.dumps(["7", "3", "99"]))
    eps = load_history(tmp_path)
    assert [e[0]["calls"][0]["tool"] for e in eps] == ["get_user_details", "get_order_details"]
    assert eps[0][0]["calls"][0]["result"] == {"a": 1}


def test_walk_finds_list_items_in_prose_results():
    text = "Found 3 packages matching 'file manager':\n\n\u2022 sfm (0.4)\n  Simple\n\u2022 walk (1.13.0)\n- yazi\n1. lf"
    assert [v for p, v in walk(text) if p == "[*]item"] == ["sfm", "walk", "yazi", "lf"]


def test_constant_arguments_are_learned_with_their_type():
    p = make(read={"nixos__nixos_info"}, tau=0.0)
    for i in range(4):
        p.learn([{"user": "info on pkg" + str(i), "calls": [
            {"tool": "nixos__nixos_search", "args": {"query": "q"}, "result": f"\u2022 pkg{i} (1.0)"},
            {"tool": "nixos__nixos_info", "args": {"name": f"pkg{i}", "type": "package", "limit": 5}, "result": "ok"}]}])
    assert p.const_for("nixos__nixos_info", "type") == "package" and p.const_for("nixos__nixos_info", "limit") == 5
    s = State()
    s.start_turn("info on new")
    s.add_call("nixos__nixos_search", "\u2022 newpkg (2.0)")
    assert p.predict(s)[0][:2] == ("nixos__nixos_info", {"name": "newpkg", "type": "package", "limit": 5})


def test_generic_user_text_patterns_capture_tickers_quotes_and_versions():
    s = State()
    s.start_turn('Compare BTC-USDT and ETH-USDT, check "neovim" at 0.10.2')
    assert s.values("user:ticker") == ["BTC-USDT", "ETH-USDT"]
    assert s.values("user:quoted") == ["neovim"] and s.values("user:version") == ["0.10.2"]
