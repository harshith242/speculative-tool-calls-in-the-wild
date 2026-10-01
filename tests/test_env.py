from fastlane import env as tau


def gold_turns(retail, task, keep):
    """A perfect episode built from the task's gold actions (only those passing `keep`)."""
    calls = []
    for a in task.evaluation_criteria.actions:
        if keep(a):
            content, error = retail.execute(a.name, a.arguments)
            calls.append({"tool": a.name, "args": a.arguments, "content": content, "error": error})
    reply = " ".join(task.evaluation_criteria.communicate_info or []) or "Done."
    return [{"user": "hi", "steps": [{"calls": calls}], "reply": reply},
            {"user": "###STOP###", "steps": [], "reply": None}]


def test_read_and_write_tools_are_tagged():
    r = tau.RetailEnv()
    assert {"get_user_details", "get_order_details", "find_user_id_by_email"} <= r.read
    assert "cancel_pending_order" in r.fns and "cancel_pending_order" not in r.read


def test_our_messages_get_tau2_reward_one_for_gold_and_zero_without_writes():
    read = tau.RetailEnv().read
    task = next(t for t in tau.get_tasks("test") if t.evaluation_criteria and t.evaluation_criteria.actions
                and any(a.name not in read for a in t.evaluation_criteria.actions))
    gold = gold_turns(tau.RetailEnv(), task, keep=lambda a: True)
    assert tau.score(task, tau.to_tau2(gold, "Hi!"), "user_stop") == 1.0
    no_writes = gold_turns(tau.RetailEnv(), task, keep=lambda a: a.name in read)
    assert tau.score(task, tau.to_tau2(no_writes, "Hi!"), "user_stop") == 0.0


def test_sampled_tasks_are_fixed_and_distinct():
    a, b = tau.sample_tasks("test", 14, 0), tau.sample_tasks("test", 14, 0)
    assert [t.id for t in a] == [t.id for t in b] and len({t.id for t in a}) == 14
    read = tau.RetailEnv().read
    assert all(any(x.name not in read for x in t.evaluation_criteria.actions) for t in a)
