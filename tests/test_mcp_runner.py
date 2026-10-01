import json
import sys
import time
from pathlib import Path

from datetime import datetime, timezone

from fastlane.mcp_runner import deepseek_peak, run_mcp_episode

CFG = {"root": str(Path(__file__).parent), "python": sys.executable,
       "servers": {"Fake Server": {"cmd": [sys.executable, "fake_mcp_server.py"], "cwd": "."}},
       "safe": {"Fake Server": ["search", "get"]}, "caps": {"predictor_per_episode": 12, "max_steps": 5}}
TASK = {"text": "Look up q and print each record.", "servers": ["Fake Server"]}
SCRIPT = '```repl\nr = fake_server__search("q")\nfor x in r:\n    print(fake_server__get(x["id"]))\n```'


class FakeAgent:
    def __init__(self, replies):
        self.replies = list(replies)

    def stream(self, messages, on_text=None):
        text = self.replies.pop(0)
        for i in range(0, len(text), 4):
            on_text(text[: i + 4])
            time.sleep(0.01)
        return {"content": text, "chunks": [[0.0, text]], "t_first": 0.0, "latency_s": 0.3, "retried": False}


def calls(trace):
    return [c for s in trace["turns"][0]["steps"] for c in s["calls"]]


def test_sptc_episode_on_real_mcp_records_measured_durations():
    trace = run_mcp_episode("t1", TASK, "B", CFG, FakeAgent([SCRIPT, "Done."]))
    gets = [c for c in calls(trace) if c["tool"] == "fake_server__get"]
    assert [c["source"] for c in gets] == ["sptc", "sptc"] and all(c["duration"] > 0.04 for c in calls(trace))
    assert "Title of q-1" in trace["turns"][0]["steps"][0]["output"] and trace["termination"] == "answered"
    assert json.loads(json.dumps(trace))["task_id"] == "t1"


def test_baseline_episode_never_speculates():
    trace = run_mcp_episode("t1", TASK, "A", CFG, FakeAgent([SCRIPT, "Done."]))
    assert trace["spec_log"] == [] and all(c["source"] == "real" for c in calls(trace))


def test_peak_window_matches_deepseek_weekday_hours():
    at = lambda d, h, m=0: datetime(2026, 10, d, h, m, tzinfo=timezone.utc)  # noqa: E731  (Oct 1 2026 is a Thursday)
    assert deepseek_peak(at(1, 8)) and deepseek_peak(at(1, 2)) and not deepseek_peak(at(1, 10))
    assert not deepseek_peak(at(1, 5)) and not deepseek_peak(at(3, 8))  # 04-06 UTC gap; Saturday
