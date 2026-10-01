"""Small shared pieces with no tau2 import: canonical call keys, JSON parsing, seeded tool latency."""
import json
import math
import random


def canonical(args):
    return json.dumps(args, sort_keys=True)


def parse_json(text):
    """Parsed JSON when the text is JSON, else the text itself."""
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


def tool_median(tool, lo, hi):
    """Each tool's median latency, log-uniform in [lo, hi] and fixed by its name."""
    u = random.Random(f"median:{tool}").random()
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def call_latency(task_id, tool, args, cfg):
    """Seconds one call takes: lognormal around the tool's median, the same in every arm and in replay."""
    rng = random.Random(f"{task_id}:{tool}:{canonical(args)}")
    return round(tool_median(tool, cfg["min_s"], cfg["max_s"]) * math.exp(rng.gauss(0, cfg["sigma"])), 3)
