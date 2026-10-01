import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastlane.mcp_env import MCPEnv

CFG = {"Fake Server": {"cmd": [sys.executable, str(Path(__file__).with_name("fake_mcp_server.py"))], "cwd": "."}}


def env():
    return MCPEnv(["Fake Server"], CFG, {"Fake Server": ["search", "get"]})


def test_tools_are_prefixed_and_whitelist_sets_read():
    e = env()
    try:
        assert {"fake_server__search", "fake_server__get", "fake_server__boom"} <= set(e.tool_specs())
        assert e.read == {"fake_server__search", "fake_server__get"}
        assert e.tool_specs()["fake_server__get"] == ["id"]
        assert "def fake_server__search(q" in e.tool_docs() and "q: string, required" in e.tool_docs()
    finally:
        e.close()


def test_execute_returns_text_and_errors():
    e = env()
    try:
        assert [r["id"] for r in json.loads(e.execute("fake_server__search", {"q": "q"})[0])] == ["q-1", "q-2"]
        content, error = e.execute("fake_server__get", {"id": "x"})
        assert not error and json.loads(content)["title"] == "Title of x"
        assert e.execute("fake_server__boom", {})[1] is True
    finally:
        e.close()


def test_concurrent_calls_overlap_on_one_session():
    e = env()
    try:
        start = time.monotonic()
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i: e.execute("fake_server__get", {"id": str(i)}), range(4)))
        assert time.monotonic() - start < 0.18  # 4 x 0.05 s calls overlapped
    finally:
        e.close()


def test_a_server_that_cannot_start_fails_fast():
    start = time.monotonic()
    try:
        MCPEnv(["Bad"], {"Bad": {"cmd": ["no-such-binary-xyz"], "cwd": "."}}, {})
        assert False, "should have raised"
    except RuntimeError:
        assert time.monotonic() - start < 10


def test_pool_spreads_calls_over_several_server_processes():
    cfg = {"Fake Server": {**CFG["Fake Server"], "pool": 3}}
    e = MCPEnv(["Fake Server"], cfg, {})
    try:
        assert len({e.execute("fake_server__pid", {})[0] for _ in range(6)}) == 3
    finally:
        e.close()


def test_warmup_call_runs_before_the_env_is_ready():
    cfg = {"Fake Server": {**CFG["Fake Server"], "pool": 2, "warmup": {"tool": "get", "args": {"id": "w"}}}}
    e = MCPEnv(["Fake Server"], cfg, {})
    try:
        start = time.monotonic()
        e.execute("fake_server__get", {"id": "x"})
        assert time.monotonic() - start < 0.5
    finally:
        e.close()
