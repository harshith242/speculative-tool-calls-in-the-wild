"""In-process code mode: RLM's persistent REPL (Repl, code_blocks, format_output) and sPTC's shadow (ShadowTurn).
The shadow runs the streaming code ahead on a deep-copied namespace so READ calls start early; the real run claims them."""
import ast
import builtins
import copy
import ctypes
import io
import operator
import queue
import re
import sys
import threading
import time
from contextlib import contextmanager
from types import FunctionType, ModuleType

from fastlane.common import parse_json

MAX_OUTPUT = 20000
STMT_BUDGET_S = 2.0  # sPTC runaway guard: pure-compute seconds per shadow statement
PURE = {"re", "json", "math", "itertools", "collections", "functools", "operator", "statistics", "string", "textwrap",
        "heapq", "bisect", "difflib", "ast", "unicodedata", "fractions", "decimal", "copy", "typing", "dataclasses"}
BLOCKED = ("open", "eval", "exec", "compile", "input", "exit", "quit", "help", "breakpoint")
COMPOUND = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try, ast.TryStar,
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)
PEEKABLE = (ast.For, ast.While, ast.If, ast.With, ast.Try)  # sPTC peeks only control flow that can fan out calls
force_ctx = threading.local()  # the shadow thread's ForceTracker


class ToolError(Exception):
    pass


def code_blocks(text):
    """Code of every ```repl or ```python block; the last may be unfinished (no closing fence yet)."""
    return [m[1] for m in re.finditer(r"```(?:repl|python)[ \t]*\n(.*?)(^```|\Z)", text, re.S | re.M)]


def format_output(stdout, stderr, ns):
    """RLM's format_execution_result, prefixed and truncated."""
    parts = ["\n" + stdout] if stdout else []
    parts += ["\n" + stderr] if stderr else []
    names = [k for k, v in ns.items() if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict, tuple))]
    parts += [f"REPL variables: {names}\n"] if names else []
    text = "REPL output:\n" + ("\n\n".join(parts) or "No output")
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + f"... + [{len(text) - MAX_OUTPUT} chars...]"


@contextmanager
def watchdog(budget, spent=lambda: 0.0):
    """Raises TimeoutError in this thread once its wall time minus spent() passes budget; yields an Event set if fired."""
    tid, t0, s0 = threading.get_ident(), time.perf_counter(), spent()
    done, fired = threading.Event(), threading.Event()

    def watch():
        while not done.wait(0.05):
            if time.perf_counter() - t0 - (spent() - s0) > budget:
                fired.set()
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(TimeoutError))
                return
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield fired
    finally:
        done.set()
        thread.join()


class Repl:
    """RLM LocalREPL: code runs in-process in one persistent namespace; each tool is a function in it."""

    def __init__(self, specs, call):
        self.specs, self.lock = specs, threading.Lock()
        self.ns = {"__name__": "__main__", "ToolError": ToolError,
                   "__builtins__": safe_builtins("the REPL")}  # tool output is untrusted: no files, no eval, pure imports
        for tool, params in specs.items():
            def hook(*args, _tool=tool, _params=params, **kw):
                content, error = call(_tool, {**dict(zip(_params, args)), **kw})
                if error:
                    raise ToolError(content)
                return parse_json(content)
            self.ns[tool] = hook

    def run(self, code, timeout=120):
        """Executes code in the namespace; returns (stdout, stderr). A bare final expression is not echoed."""
        out, err = io.StringIO(), io.StringIO()
        with self.lock:  # stdout and stderr are process-wide, so runs are serialized
            old = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = out, err
            try:
                with watchdog(timeout) as fired:
                    try:
                        exec(code, self.ns)
                    except Exception as e:
                        err.write(f"\n{type(e).__name__}: {f'the code ran over {timeout}s' if fired.is_set() else e}")
            finally:
                sys.stdout, sys.stderr = old
        return out.getvalue(), err.getvalue()


class SpecValue:
    """A tool result still in flight; any use of it waits for the call and decodes it like the real hook."""
    __slots__ = ("_fut", "_val")

    def __init__(self, fut):
        self._fut, self._val = fut, _UNSET

    def _force(self):
        if self._val is _UNSET:
            tracker = getattr(force_ctx, "tracker", None)
            t0 = time.perf_counter()
            tracker and tracker.begin(t0)
            try:
                content, error = self._fut.result()
            finally:
                tracker and tracker.end(time.perf_counter() - t0)
            if error:
                raise ToolError(content)
            self._val = parse_json(content)
        return self._val

    def __getattr__(self, name):
        return getattr(self._force(), name)

    def __deepcopy__(self, memo):
        return self


_UNSET = object()
_UNARY = {"__str__": str, "__repr__": repr, "__bool__": bool, "__len__": len, "__iter__": iter, "__int__": int,
          "__float__": float, "__hash__": hash}
_BINARY = {"__format__": format, "__contains__": lambda v, x: x in v, "__getitem__": operator.getitem,
           "__eq__": operator.eq, "__ne__": operator.ne, "__lt__": operator.lt, "__le__": operator.le,
           "__gt__": operator.gt, "__ge__": operator.ge, "__add__": operator.add, "__mul__": operator.mul,
           "__radd__": lambda v, o: o + v}
for _name, _fn in _UNARY.items():
    setattr(SpecValue, _name, lambda self, _fn=_fn: _fn(self._force()))
for _name, _fn in _BINARY.items():
    setattr(SpecValue, _name, lambda self, o, _fn=_fn: _fn(self._force(), deep_force(o)))


class ForceTracker:
    """Seconds the shadow thread spent waiting on tool results, so the runaway guard counts only compute."""

    def __init__(self):
        self.total, self.since = 0.0, None

    def begin(self, now):
        self.since = now

    def end(self, waited):
        self.total, self.since = self.total + waited, None

    def seconds(self):
        return self.total + (time.perf_counter() - self.since if self.since else 0.0)


class NonSpeculated:
    """Marker for the result of a call the shadow must not run (a write); statements that read it are skipped."""

    def __init__(self, tool):
        self.tool = tool


class Opaque:
    """Marker for a value that refused deepcopy; any use aborts the shadow."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, k=None):
        raise RuntimeError(f"opaque value {self.name!r} touched in shadow")

    __getitem__ = __iter__ = __getattr__


def deep_force(obj, depth=3):
    """Replaces SpecValues by their results through lists, tuples and dicts."""
    if isinstance(obj, SpecValue):
        return obj._force()
    if depth and isinstance(obj, (list, tuple)):
        return type(obj)(deep_force(x, depth - 1) for x in obj)
    if depth and isinstance(obj, dict):
        return {k: deep_force(v, depth - 1) for k, v in obj.items()}
    return obj


def force_in_place(obj, depth=3):
    """Forces SpecValues inside lists and dicts in place, so the namespace objects keep their identity."""
    items = enumerate(obj) if isinstance(obj, list) else list(obj.items()) if isinstance(obj, dict) else ()
    for k, v in items:
        if isinstance(v, SpecValue):
            obj[k] = v._force()
        elif depth > 1:
            force_in_place(v, depth - 1)


def contains_nonspec(obj, depth=3):
    if isinstance(obj, NonSpeculated):
        return True
    items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else ()
    return depth > 0 and any(contains_nonspec(x, depth - 1) for x in items)


def _copy(name, v):
    if isinstance(v, ModuleType):
        return v if v.__name__.split(".")[0] in PURE else Opaque(name)
    try:
        return copy.deepcopy(v)
    except Exception:
        try:  # a plain cast keeps the data of a dict or list with an un-copyable attribute
            if isinstance(v, dict):
                return {k: copy.deepcopy(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return (list if isinstance(v, list) else tuple)(copy.deepcopy(x) for x in v)
        except Exception:
            pass
        return Opaque(name)


def snapshot(src, skip):
    """sPTC snapshot_ns: deep copy of a namespace without dunders and `skip` names; its functions resolve globals in the copy."""
    out = {k: _copy(k, v) for k, v in src.items() if not k.startswith("__") and k not in skip}
    for k, v in out.items():
        if isinstance(v, FunctionType) and v.__globals__ is src:
            fn = FunctionType(v.__code__, out, v.__name__, v.__defaults__, v.__closure__)
            fn.__dict__.update(v.__dict__)
            fn.__kwdefaults__ = v.__kwdefaults__
            out[k] = fn
    return out


def safe_builtins(where):
    """Builtins without file, eval/exec or interactive access, and imports limited to pure stdlib modules."""
    b = dict(vars(builtins))
    for name in BLOCKED + ("globals", "locals"):
        b[name] = lambda *a, _n=name, **k: (_ for _ in ()).throw(RuntimeError(f"{_n}() blocked in {where}"))

    def pure_import(name, *a, **k):
        if name.split(".")[0] not in PURE:
            raise RuntimeError(f"import {name!r} blocked in {where}")
        return __import__(name, *a, **k)
    b["__import__"] = pure_import
    return b


def shadow_builtins():
    b = safe_builtins("shadow")
    b["print"] = lambda *a, **k: None  # output is discarded, so it never forces a wait
    return b


def _names(node, ctx):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


def _bound(node):
    """Names a statement binds, minus comprehension and lambda locals (they never leak)."""
    local = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            local |= {x.id for g in n.generators for x in ast.walk(g.target) if isinstance(x, ast.Name)}
        elif isinstance(n, ast.Lambda):
            local |= {a.arg for a in n.args.args}
    defs = {n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return (_names(node, (ast.Store, ast.Del)) | defs) - local


def _parse_prefix(text):
    """AST of the longest parseable prefix of lines, and the lines."""
    lines = text.split("\n")
    for end in range(len(lines), 0, -1):
        try:
            return ast.parse("\n".join(lines[:end])), lines
        except SyntaxError:
            pass
    return ast.Module([], []), lines


class ShadowTurn:
    """sPTC shadow for one reply: runs closed statements on a copy of repl.ns, so READ calls start while the model writes."""

    def __init__(self, repl, speculate, writes):
        self.speculate, self.writes, self.specs, self.code = speculate, set(writes), repl.specs, ""
        self.tracker, self.builtins = ForceTracker(), shadow_builtins()
        self.hooks = {tool: self._hook(tool) for tool in self.specs}
        self.ns = self._fork(repl.ns)
        self.q, self.done, self.aborted = queue.Queue(), threading.Event(), None
        self.sent, self.peeked = 0, ""  # statements queued so far; tail text of the last peek
        threading.Thread(target=self._work, daemon=True, name="shadow").start()

    def _hook(self, tool):
        def hook(*args, **kw):
            if tool in self.writes:
                return NonSpeculated(tool)
            params = deep_force({**dict(zip(self.specs[tool], args)), **kw})
            if contains_nonspec(params):
                raise RuntimeError(f"{tool} args depend on a non-speculated result")
            fut = self.speculate(tool, params)
            return NonSpeculated(tool) if fut is None else SpecValue(fut)
        return hook

    def _fork(self, src):
        ns = snapshot(src, self.hooks)
        ns.update(self.hooks, __builtins__=self.builtins, __name__="__main__")
        return ns

    def feed(self, code):
        self.code = code
        self._emit(code[:code.rfind("\n") + 1], final=False)

    def _emit(self, text, final):
        tree, lines = _parse_prefix(text)
        tail = tree.body[-1] if tree.body and not final and isinstance(tree.body[-1], COMPOUND) else None
        closed = tree.body[:len(tree.body) - (tail is not None)]
        for node in closed[self.sent:]:
            self.q.put(("run", node))
        self.sent = max(self.sent, len(closed))
        if isinstance(tail, PEEKABLE):  # an unfinished loop may already show the calls it will make
            src = "\n".join(lines[tail.lineno - 1:tail.end_lineno])
            new = src[len(self.peeked):] if src.startswith(self.peeked) else src
            if any(tool in new for tool in self.specs):
                self.q.put(("peek", tail))
                self.peeked = src

    def end(self, timeout=600):
        self._emit(self.code, final=True)
        self.q.put(None)
        self.done.wait(timeout)
        self.aborted = self.aborted or "turn_end"

    def _work(self):
        force_ctx.tracker = self.tracker
        try:
            while (item := self.q.get()) is not None:
                kind, node = item
                if self.aborted:
                    continue
                try:
                    why = self._exec(node, self._fork(self.ns) if kind == "peek" else self.ns)
                except BaseException as e:
                    why = f"{type(e).__name__}: {e}"
                if kind == "run":
                    self.aborted = why
        finally:
            self.done.set()

    def _exec(self, node, ns):
        """Runs one statement in ns; returns why the shadow must stop, or None."""
        bound, reads = _bound(node), _names(node, ast.Load)
        if bound & self.hooks.keys():
            return "rebinds a tool"
        tainted = [n for n in reads if contains_nonspec(ns.get(n))]
        if tainted:  # reads a write's result: skip it and poison what it would have bound
            ns.update({n: NonSpeculated("tainted:" + "+".join(tainted)) for n in bound})
            return None
        try:
            for name in reads:
                if isinstance(ns.get(name), SpecValue):
                    ns[name] = ns[name]._force()
                else:
                    force_in_place(ns.get(name))
            with watchdog(STMT_BUDGET_S, self.tracker.seconds):
                exec(compile(ast.Module([node], []), "<shadow>", "exec"), ns)
        except BaseException as e:
            return f"{type(e).__name__}: {e}"
        return None
