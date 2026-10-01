"""Promise store: every tool call, real or speculative, goes through one table keyed by (tool, canonical args).
A speculative result is used at most once; any real non-READ call drops all pending results."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastlane.common import canonical


def failure_kind(content, error):
    """None for success; else rate_limit, timeout, transport or tool_error."""
    if not error:
        return None
    text = str(content).lower()
    for kind, words in (("rate_limit", ("429", "rate limit", "too many")), ("timeout", ("timeout", "timed out")),
                        ("transport", ("connection", "transport", "closed", "unreachable"))):
        if any(w in text for w in words):
            return kind
    return "tool_error"


class Store:
    def __init__(self, execute, latency, read, clock=time.monotonic, cap=None):
        """latency(tool, args) -> injected seconds, or None for real tools; cap limits non-sPTC speculative launches."""
        self.execute, self.latency, self.read, self.clock, self.cap = execute, latency, read, clock, cap
        self.pool = ThreadPoolExecutor(max_workers=32)
        self.lock = threading.Lock()
        self.pending = {}  # (tool, canonical args) -> speculative entry not used yet
        self.log = []  # every speculative launch, for waste and leakage
        self.on_result = None  # runner hook, called after each real call

    def _run(self, tool, args):
        content, error = self.execute(tool, args)
        if self.latency:
            time.sleep(self.latency(tool, args))
        return content, error

    def _spec(self, entry):
        content, error = self._run(entry["tool"], entry["args"])
        entry["t_end"], entry["failure"] = self.clock(), failure_kind(content, error)
        return content, error

    def speculate(self, tool, args, source):
        """Start a READ call early unless the same call is already pending; returns its future."""
        if tool not in self.read:
            return None
        key = (tool, canonical(args))
        with self.lock:
            if key not in self.pending:
                if self.cap is not None and source != "sptc" and sum(e["source"] != "sptc" for e in self.log) >= self.cap:
                    return None
                entry = {"tool": tool, "args": args, "source": source, "t_launch": self.clock(), "used": False}
                entry["future"] = self.pool.submit(self._spec, entry)
                self.pending[key] = entry
                self.log.append(entry)
            return self.pending[key]["future"]

    def evict(self, source):
        """Drop pending results from one source (sPTC evicts its unclaimed speculations at the end of each cell)."""
        with self.lock:
            self.pending = {k: e for k, e in self.pending.items() if e["source"] != source}

    def call(self, tool, args):
        """A real call: takes the pending speculative result if there is one."""
        t_request = self.clock()
        with self.lock:
            if tool not in self.read:
                self.pending.clear()
            entry = self.pending.pop((tool, canonical(args)), None)
        if entry:
            entry["used"] = True
            content, error = entry["future"].result()
            source, t_launch, duration = entry["source"], entry["t_launch"], entry["t_end"] - entry["t_launch"]
        else:
            content, error = self._run(tool, args)
            source, t_launch, duration = "real", t_request, self.clock() - t_request
        info = {"tool": tool, "args": args, "content": content, "error": error, "read": tool in self.read,
                "source": source, "t_request": t_request, "t_launch": t_launch, "t_done": self.clock(),
                "duration": round(duration, 3), "latency": self.latency(tool, args) if self.latency else round(duration, 3)}
        if self.on_result:
            self.on_result(info)
        return content, error, info
