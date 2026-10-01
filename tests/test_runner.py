import re
import time

from fastlane import env as tau
from fastlane.predictor import USER_PATTERNS, Predictor
from fastlane.runner import run_episode

CFG = {"latency": {"min_s": 0.05, "max_s": 0.1, "sigma": 0.1}}


class FakeAgent:
    def __init__(self, replies):
        self.replies = list(replies)

    def stream(self, messages, on_text=None):
        text = self.replies.pop(0)
        for i in range(0, len(text), 4):
            on_text(text[: i + 4])
            time.sleep(0.03)
        return {"content": text, "chunks": [[0.0, text]], "t_first": 0.0, "latency_s": 0.0, "retried": False}


class FakeUser:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages):
        return {"content": self.replies.pop(0)}


def user_id():
    return next(u for u in tau.RetailEnv().env.tools.db.users if re.fullmatch(USER_PATTERNS["user_id"], u))


def episode(arm, predictor=None):
    u = user_id()
    script = f'```repl\nu = get_user_details("{u}")\nprint(len(u["orders"]))\n```'
    task = tau.get_tasks("test")[0]
    return run_episode(task, arm, CFG, FakeAgent([script, "Done."]), FakeUser([f"My id is {u}", "###STOP###"]), predictor)


def test_sptc_arm_starts_the_call_while_the_code_streams():
    trace = episode("B")
    step = trace["turns"][0]["steps"][0]
    assert step["calls"][0]["tool"] == "get_user_details" and step["calls"][0]["source"] == "sptc"
    assert step["sptc"][0]["tool"] == "get_user_details" and step["calls"][0]["t_launch"] < step["calls"][0]["t_request"]
    assert trace["termination"] == "user_stop" and trace["reward"] in (0.0, 1.0)


def test_predictor_arm_starts_the_call_before_the_model_writes_it():
    pred = Predictor(tau.RetailEnv().read, k=3, tau=0.3, min_obs=3, min_precision=0.6, min_avail=3)
    for i in range(3):
        uid = f"x_y_{1000 + i}"
        pred.learn([{"user": f"My id is {uid}", "calls": [{"tool": "get_user_details", "args": {"user_id": uid}, "result": {}}]}])
    trace = episode("C", pred)
    assert trace["turns"][0]["steps"][0]["calls"][0]["source"] == "pred"
    assert trace["spec_log"][0]["source"] == "pred" and trace["spec_log"][0]["used"] is True


def test_baseline_arm_never_speculates():
    trace = episode("A")
    assert trace["spec_log"] == [] and trace["turns"][0]["steps"][0]["calls"][0]["source"] == "real"
