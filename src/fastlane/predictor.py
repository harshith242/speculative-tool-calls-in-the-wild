"""Trajectory predictor: a counting table over past episodes (no ML, no LLM).
It counts which tool follows the last 2 / 1 / 0 tools and where each argument's value came from,
then guesses (tool, args) calls whose score p(tool) * precision(source) clears tau."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from fastlane.common import canonical, parse_json

TURN, REPLY = "<turn>", "<reply>"
USER_PATTERNS = {"email": r"[\w.+-]+@[\w-]+\.[\w.]+", "order": r"#W\d{7}", "user_id": r"\b[a-z]+_[a-z]+_\d{4}\b",
                 "zip": r"\b\d{5}\b", "ticker": r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})+\b", "quoted": r'"([^"\n]{2,60})"',
                 "version": r"\b\d+\.\d+(?:\.\d+)?\b"}


LIST_ITEM = re.compile(r"^[ \t]*(?:[\u2022*-]|\d+[.)])[ \t]+([\w.@+/-]+)", re.M)


def walk(value, path=""):
    """(path, leaf string) pairs; list items share a [*] path and ID-like dict keys share a * path."""
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k)[:1].isdigit() or str(k).startswith("#"):
                yield f"{path}.*key", str(k)
                yield from walk(v, f"{path}.*")
            else:
                yield from walk(v, f"{path}.{k}")
    elif isinstance(value, list):
        for v in value:
            yield from walk(v, f"{path}[*]")
    elif value is not None:
        yield path, str(value)
        if isinstance(value, str):  # prose results: each list line's first token is a candidate value
            for item in LIST_ITEM.findall(value):
                yield f"{path}[*]item", item


def contexts(tokens):
    """Back-off contexts, longest first."""
    return list(dict.fromkeys([tuple(tokens[-2:]), tuple(tokens[-1:]), ()]))


class State:
    """What the predictor sees mid-episode: token history, latest user message, each tool's latest result."""

    def __init__(self):
        self.tokens, self.user, self.results, self.text, self.turn_text = [], "", {}, "", ""

    def start_turn(self, user):
        self.tokens.append(TURN)
        self.user, self.turn_text = user, user
        self.text += "\n" + user

    def add_call(self, tool, result):
        self.tokens.append(tool)
        self.results[tool] = result
        dump = result if isinstance(result, str) else json.dumps(result)
        self.text += "\n" + dump
        self.turn_text += "\n" + dump

    def values(self, source):
        """Values a source resolves to now; several values mean a fan-out."""
        kind, _, spec = source.partition(":")
        if kind == "const":  # a literal the agent keeps passing, kept with its JSON type
            return [json.loads(spec)]
        if kind == "user":
            return list(dict.fromkeys(re.findall(USER_PATTERNS[spec], self.user)))
        tool, _, path = spec.partition("|")
        if tool not in self.results:
            return []
        return list(dict.fromkeys(v for p, v in walk(self.results[tool]) if p == path))


class Predictor:
    def __init__(self, read, k=3, tau=0.3, min_obs=3, min_precision=0.6, min_avail=3):
        self.read, self.k, self.tau = set(read), k, tau
        self.min_obs, self.min_precision, self.min_avail = min_obs, min_precision, min_avail
        self.next = defaultdict(Counter)  # context -> Counter(next token)
        self.sources = defaultdict(set)  # (tool, arg) -> sources that ever held the real value
        self.avail, self.match = Counter(), Counter()  # (tool, arg, source) -> times available / times it held the value
        self.arg_names = {}

    def learn(self, episode):
        """episode: [{"user": str, "calls": [{"tool", "args", "result"}]}]"""
        state = State()
        for turn in episode:
            state.start_turn(turn["user"])
            for call in turn["calls"]:
                for ctx in contexts(state.tokens):
                    self.next[ctx][call["tool"]] += 1
                self.arg_names[call["tool"]] = list(call["args"])
                for arg, value in call["args"].items():
                    self._count_source(state, call["tool"], arg, value)
                state.add_call(call["tool"], call["result"])
            for ctx in contexts(state.tokens):
                self.next[ctx][REPLY] += 1

    def _count_source(self, state, tool, arg, raw):
        value = str(raw)
        found = {f"user:{k}" for k in USER_PATTERNS if value in state.values(f"user:{k}")}
        found |= {f"res:{t}|{p}" for t, r in state.results.items() for p, v in walk(r) if v == value}
        found.add("const:" + json.dumps(raw, sort_keys=True))
        self.sources[(tool, arg)] |= found
        for src in self.sources[(tool, arg)]:
            vals = state.values(src)
            if vals:
                self.avail[(tool, arg, src)] += 1
                self.match[(tool, arg, src)] += value in vals or raw in vals

    def const_for(self, tool, arg):
        """The literal this argument usually takes (a valid const source), else None."""
        best = max(((self.precision(tool, arg, s), s) for s in self.sources[(tool, arg)] if s.startswith("const:")), default=None)
        return json.loads(best[1][6:]) if best and best[0] >= self.min_precision else None

    def precision(self, tool, arg, src):
        n = self.avail[(tool, arg, src)]
        return self.match[(tool, arg, src)] / n if n >= self.min_avail else 0.0

    def next_probs(self, tokens):
        for ctx in contexts(tokens):
            counts = self.next.get(ctx)
            if counts and sum(counts.values()) >= self.min_obs:
                total = sum(counts.values())
                return {t: c / total for t, c in counts.items()}
        return {}

    def seen(self, tokens, tool):
        """Whether this (last token -> tool) transition was counted before: the seen-before split."""
        return self.next.get(tuple(tokens[-1:]), {}).get(tool, 0) > 0

    def _fill(self, state, tool):
        """(argument sets, min source precision); each arg uses its best valid source, and lists fan out."""
        best = {}
        for arg in self.arg_names.get(tool, []):
            ranked = sorted((self.precision(tool, arg, s), s) for s in self.sources[(tool, arg)])
            ranked = [(q, s) for q, s in reversed(ranked) if q >= self.min_precision and state.values(s)]
            if not ranked:
                return None
            best[arg] = ranked[0]
        arg_sets = [{}]
        for arg, (_, src) in best.items():
            arg_sets = [{**a, arg: v} for a in arg_sets for v in state.values(src)]
        return arg_sets[: self.k], min((q for q, _ in best.values()), default=1.0)

    def predict(self, state, tau=None, k=None):
        """Up to k (tool, args, score) guesses with score >= tau, best first."""
        tau = self.tau if tau is None else tau
        guesses = []
        for tool, p in self.next_probs(state.tokens).items():
            if tool not in self.read or p < tau:
                continue
            filled = self._fill(state, tool)
            if filled:
                guesses += [(tool, args, p * filled[1]) for args in filled[0]]
        guesses = sorted((g for g in guesses if g[2] >= tau), key=lambda g: -g[2])
        return guesses[: k or self.k]


def from_trace(trace):
    """Our recorded trace -> the predictor's episode format (real calls only)."""
    return [{"user": t["user"], "calls": [{"tool": c["tool"], "args": c["args"], "result": parse_json(c["content"])}
                                          for s in t["steps"] for c in s["calls"]]} for t in trace["turns"]]


def precision_at(pred, episodes, tau):
    """Share of guesses (fired at turn start and after each call) that match a later real call in the same turn."""
    hits = total = 0
    for ep in episodes:
        state = State()
        for turn in ep:
            state.start_turn(turn["user"])
            later = [(c["tool"], canonical(c["args"])) for c in turn["calls"]]
            for i in range(len(turn["calls"]) + 1):
                for tool, args, _ in pred.predict(state, tau=tau):
                    total += 1
                    hits += (tool, canonical(args)) in later[i:]
                if i < len(turn["calls"]):
                    state.add_call(turn["calls"][i]["tool"], turn["calls"][i]["result"])
    return (hits / total if total else 0.0), total


def choose_tau(make, episodes, target, grid):
    """Smallest tau whose precision on the last 20% of episodes reaches target, learning from the first 80%."""
    cut = int(len(episodes) * 0.8)
    pred = make(0.0)
    for ep in episodes[:cut]:
        pred.learn(ep)
    for tau in sorted(grid):
        prec, n = precision_at(pred, episodes[cut:], tau)
        if n and prec >= target:
            return tau
    return max(grid)


def load_history(runs):
    """Our arm-A episodes on train-split tasks, in sampled order, as predictor episodes."""
    folder = Path(runs) / "history"
    order = json.loads((folder / "order.json").read_text()) if (folder / "order.json").exists() else []
    return [from_trace(json.loads((folder / f"{t}.json").read_text())) for t in order if (folder / f"{t}.json").exists()]


def warm_predictor(cfg, read, tau=None):
    """Predictor trained on the history episodes; tau is picked on them unless given."""
    p = cfg["predictor"]

    def make(t):
        return Predictor(read, k=p["k"], tau=t, min_obs=p["min_obs"], min_precision=p["min_precision"],
                         min_avail=p["min_avail"])
    episodes = load_history(cfg["paths"]["runs"])
    if tau is None:
        tau = choose_tau(make, episodes, p["target_precision"], cfg["replay"]["tau_grid"])
    pred = make(tau)
    for ep in episodes:
        pred.learn(ep)
    return pred
