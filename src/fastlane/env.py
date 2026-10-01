"""Everything that touches tau2: retail tools (READ vs the rest), user-simulator prompt, tasks, and the reward.
Our episodes are converted to tau2 messages and scored by tau2's DB-state evaluator (NL assertions are skipped)."""
import inspect
import os
import random
import threading
from pathlib import Path

os.environ.setdefault("TAU2_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402
from tau2.environment.toolkit import ToolType  # noqa: E402
from tau2.evaluator.evaluator_env import EnvironmentEvaluator  # noqa: E402
from tau2.user.user_simulator import SYSTEM_PROMPT, get_global_user_sim_guidelines  # noqa: E402
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER  # noqa: E402

STOP_TOKENS = (STOP, TRANSFER, OUT_OF_SCOPE)


class RetailEnv:
    """A fresh retail DB per episode; tau2 calls are serialized because speculative reads run on other threads."""

    def __init__(self):
        self.env = get_environment()
        self.fns = self.env.tools.tools
        self.read = {n for n in self.fns if self.env.tools.tool_type(n) == ToolType.READ}
        self.lock = threading.Lock()

    def execute(self, tool, args):
        with self.lock:
            msg = self.env.get_response(ToolCall(id="fl", name=tool, arguments=args, requestor="assistant"))
        return msg.content, msg.error

    def tool_specs(self):
        """{tool: [param names]} for the REPL tool functions."""
        return {n: list(inspect.signature(f).parameters) for n, f in self.fns.items()}

    def tool_docs(self):
        return "\n\n".join(f'def {n}{inspect.signature(f)}:\n    """{inspect.getdoc(f)}"""' for n, f in self.fns.items())


def user_system_prompt(task):
    return SYSTEM_PROMPT.format(global_user_sim_guidelines_with_persona=get_global_user_sim_guidelines(),
                                instructions=str(task.user_scenario))


def sample_tasks(split, n, seed):
    """n tasks from a tau2 split whose gold actions write to the DB (else any episode scores 1), fixed by seed."""
    kit = get_environment().tools
    writes = {t for t in kit.tools if kit.tool_type(t) == ToolType.WRITE}
    tasks = [t for t in get_tasks(split)
             if t.evaluation_criteria and any(a.name in writes for a in t.evaluation_criteria.actions or [])]
    return random.Random(seed).sample(tasks, n)


def to_tau2(turns, greeting):
    """Our turns -> tau2 messages: each real tool call becomes an assistant tool-call message plus its tool reply."""
    msgs, n = [AssistantMessage(role="assistant", content=greeting)], 0
    for turn in turns:
        msgs.append(UserMessage(role="user", content=turn["user"]))
        for step in turn["steps"]:
            for c in step["calls"]:
                n += 1
                call = ToolCall(id=f"c{n}", name=c["tool"], arguments=c["args"], requestor="assistant")
                msgs.append(AssistantMessage(role="assistant", tool_calls=[call]))
                msgs.append(ToolMessage(id=f"c{n}", role="tool", content=c["content"], requestor="assistant",
                                        error=c["error"]))
        if turn.get("reply") is not None:
            msgs.append(AssistantMessage(role="assistant", content=turn["reply"]))
    return msgs


def score(task, messages, termination):
    """tau2's DB-state reward (0/1): the final DB must match the gold actions' DB; an unfinished episode scores 0."""
    if termination != "user_stop":
        return 0.0
    return EnvironmentEvaluator.calculate_reward(environment_constructor=get_environment, task=task,
                                                 full_trajectory=messages).reward
