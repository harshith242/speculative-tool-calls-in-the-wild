import json
import time

from fastlane.repl import Repl, ShadowTurn, ToolError, code_blocks, format_output
from fastlane.store import Store

SPECS = {"get_user_details": ["user_id"], "get_order_details": ["order_id"],
         "cancel_pending_order": ["order_id", "reason"]}
READ = {"get_user_details", "get_order_details"}
WRITES = {"cancel_pending_order"}


def make(latency=0.1):
    calls = []

    def execute(tool, args):
        calls.append((tool, args))
        if tool == "get_user_details":
            return json.dumps({"orders": ["#W1", "#W2"]}), False
        return json.dumps({"id": args.get("order_id")}), False
    store = Store(execute, lambda tool, args: latency, READ)
    repl = Repl(SPECS, lambda tool, args: store.call(tool, args)[:2])
    return store, repl, calls


def shadow(store, repl):
    return ShadowTurn(repl, lambda tool, args: store.speculate(tool, args, "sptc"), WRITES)


def launched(store):
    return [(e["tool"], e["args"]) for e in store.log]


def wait_for(cond, limit=0.5):
    end = time.monotonic() + limit
    while time.monotonic() < end and not cond():
        time.sleep(0.01)
    return cond()


def test_repl_persists_variables_and_captures_print():
    _, repl, _ = make(0)
    repl.run("x = 41")
    assert repl.run("print(x + 1)") == ("42\n", "")
    assert repl.run("x + 1") == ("", "")  # a bare final expression is not echoed


def test_repl_tool_hook_maps_positional_args_and_decodes_json():
    _, repl, calls = make(0)
    out, err = repl.run("u = get_user_details('u1')\nprint(u['orders'])")
    assert (out, err) == ("['#W1', '#W2']\n", "") and calls == [("get_user_details", {"user_id": "u1"})]


def test_repl_tool_error_is_catchable_and_uncaught_errors_go_to_stderr():
    repl = Repl(SPECS, lambda tool, args: ("no such order", True))
    out, _ = repl.run("try:\n    get_order_details('#W0')\nexcept ToolError as e:\n    print('caught', e)")
    assert out == "caught no such order\n"
    assert repl.run("get_order_details('#W0')")[1] == "\nToolError: no such order"
    assert repl.run("print(undefined)")[1] == "\nNameError: name 'undefined' is not defined"
    assert issubclass(ToolError, Exception)


def test_repl_timeout_interrupts_runaway_code():
    _, repl, _ = make(0)
    start = time.monotonic()
    _, err = repl.run("while True: pass", timeout=0.5)
    assert "TimeoutError" in err and time.monotonic() - start < 2


def test_format_output_cases():
    assert format_output("", "", {}) == "REPL output:\nNo output"
    ns = {"n": 1, "s": "x", "_hidden": 2, "f": print, "ToolError": ToolError, "__name__": "m"}
    assert format_output("out\n", "err", ns) == "REPL output:\n\nout\n\n\n\nerr\n\nREPL variables: ['n', 's']\n"
    text = format_output("x" * 30000, "", {})
    cut = len("REPL output:\n\n" + "x" * 30000) - 20000
    assert text == ("REPL output:\n\n" + "x" * 30000)[:20000] + f"... + [{cut} chars...]"


def test_code_blocks():
    assert code_blocks("no code here") == []
    assert code_blocks("a\n```repl\nx = 1\n```\nb\n```python\ny = 2\n```\n") == ["x = 1\n", "y = 2\n"]
    assert code_blocks("```repl\nx = 1\n```\n```repl\ny = get(") == ["x = 1\n", "y = get("]
    assert code_blocks("```bash\nls\n```") == []


def test_shadow_runs_independent_loop_calls_in_parallel():
    store, repl, _ = make()
    turn = shadow(store, repl)
    turn.feed('for o in ["#W1","#W2","#W3"]:\n    print(get_order_details(o))\n')
    turn.end()
    times = [e["t_launch"] for e in store.log]
    assert len(times) == 3 and max(times) - min(times) < 0.15


def test_shadow_follows_a_chain_of_dependent_calls():
    store, repl, _ = make()
    turn = shadow(store, repl)
    turn.feed('x = get_user_details("u1")\nfor o in x["orders"]:\n    get_order_details(o)\n')
    turn.end()
    assert launched(store) == [("get_user_details", {"user_id": "u1"}), ("get_order_details", {"order_id": "#W1"}),
                               ("get_order_details", {"order_id": "#W2"})]


def test_shadow_skips_statements_that_read_a_write_result_but_continues():
    store, repl, calls = make(0.01)
    turn = shadow(store, repl)
    turn.feed('r = cancel_pending_order("#W1", "no longer needed")\ny = get_order_details(r["id"])\n'
              'z = get_order_details("#W9")\n')
    turn.end()
    store.log[0]["future"].result()
    assert launched(store) == [("get_order_details", {"order_id": "#W9"})]
    assert [c[0] for c in calls] == ["get_order_details"] and turn.aborted == "turn_end"


def test_shadow_namespace_is_a_copy():
    store, repl, _ = make(0)
    repl.run("data = {'k': 1}")
    turn = shadow(store, repl)
    turn.feed("data['k'] = 2\nnewvar = 5\n")
    turn.end()
    assert turn.ns["data"] == {"k": 2} and turn.ns["newvar"] == 5
    assert repl.ns["data"] == {"k": 1} and "newvar" not in repl.ns


def test_shadow_emits_only_closed_statements_before_end():
    store, repl, _ = make()
    turn = shadow(store, repl)
    turn.feed('a = get_order_details("#W1")\nb = get_order_details("#W2"')
    assert wait_for(lambda: len(store.log) == 1) and launched(store)[0][1] == {"order_id": "#W1"}
    time.sleep(0.1)
    assert len(store.log) == 1  # the unfinished call is not launched
    turn.feed('a = get_order_details("#W1")\nb = get_order_details("#W2")\n')
    assert wait_for(lambda: len(store.log) == 2)  # a finished simple statement launches before end()
    turn.end()
    assert len(store.log) == 2


def test_shadow_peeks_into_an_unfinished_loop():
    store, repl, _ = make()
    turn = shadow(store, repl)
    turn.feed('for o in ["#W1","#W2"]:\n    get_order_details(o)\n')
    assert wait_for(lambda: len(store.log) == 2)  # no end(): the loop is still open
    turn.end()
    assert len(store.log) == 2


def test_repl_blocks_files_eval_and_unsafe_imports_but_allows_pure_ones():
    r = Repl({}, lambda tool, args: ("{}", False))
    for code in ('open(".env").read()', 'import os', 'import subprocess', 'eval("1")', 'from pathlib import Path'):
        assert "blocked in the REPL" in r.run(code)[1], code
    assert r.run('import json, re, math\nprint(json.dumps({"a": math.floor(1.5)}))') == ('{"a": 1}\n', "")
