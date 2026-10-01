"""MCP servers as the agent's tools: one stdio session per server, kept open on a background asyncio loop.
Tool names are prefixed by server (wikipedia__get_article) so tools of different servers never collide."""
import asyncio
import json
import os
import re
import threading
import time
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CALL_TIMEOUT = 60


def slug(server):
    return re.sub(r"\W+", "_", server.lower()).strip("_")


def result_text(r):
    """One text per call: structured content if given, several JSON blocks merged into a list, else joined text."""
    structured = getattr(r, "structured_content", None) or getattr(r, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured["result"] if isinstance(structured, dict) and set(structured) == {"result"} else structured)
    texts = [getattr(c, "text", "") for c in r.content]
    try:
        return json.dumps([json.loads(t) for t in texts]) if len(texts) > 1 else "\n".join(texts)
    except json.JSONDecodeError:
        return "\n".join(texts)


def _param(name, spec, required):
    kind = spec.get("type") or "|".join(str(x.get("type", "any")) for x in spec.get("anyOf", [])) or "any"
    bits = [f"{name}: {kind}"]
    bits += [f"one of {spec['enum']}"] if "enum" in spec else []
    bits += [f"default {spec['default']!r}"] if "default" in spec else []
    bits += ["required"] if name in required else []
    return "        " + ", ".join(bits) + (f" - {spec['description'].strip()}" if spec.get("description") else "")


def _doc(name, tool):
    schema = _schema(tool)
    props, required = schema.get("properties", {}), set(schema.get("required", []))
    args = "\n".join(_param(p, s, required) for p, s in props.items())
    return f'def {name}({", ".join(props)}):\n    """{(tool.description or "").strip()}\n    Args:\n{args}"""'


def _schema(tool):
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}


class MCPEnv:
    def __init__(self, servers, cfg, safe, python="python"):
        """servers: names to start; cfg: {name: {cmd, cwd}}; safe: {name: [tool]} the speculation_safe whitelist."""
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.sessions, self.tools, self.ready = {}, {}, threading.Event()
        self.lock, self.turn = threading.Lock(), 0
        python = os.path.abspath(python) if os.sep in python else python  # servers run in their own cwd
        self.keeper = asyncio.run_coroutine_threadsafe(self._keep(servers, cfg, python), self.loop)
        deadline = time.monotonic() + 120
        while not self.ready.wait(0.1):  # fail fast if a server dies instead of waiting out the timeout
            if self.keeper.done() or time.monotonic() > deadline:
                raise RuntimeError(f"MCP servers failed to start: {self.keeper.exception() if self.keeper.done() else 'timeout'}")
        self.read = {n for n, (s, t) in self.tools.items() if t.name in safe.get(s, [])}

    async def _keep(self, servers, cfg, python):
        self.stop = asyncio.Event()
        async with AsyncExitStack() as stack:  # opened and closed in this one task, as anyio requires
            for s in servers:
                cmd = [python if c == "{python}" else c for c in cfg[s]["cmd"]]
                cmd[0] = os.path.abspath(cmd[0]) if os.sep in cmd[0] else cmd[0]  # a relative interpreter path breaks under cwd
                params = StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=cfg[s]["cwd"])
                self.sessions[s] = []
                for _ in range(cfg[s].get("pool", 1)):  # several copies of a stateless server that handles calls serially
                    read, write = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    self.sessions[s].append(session)
                if cfg[s].get("warmup"):  # pay each copy's first-call setup, in parallel, before the episode clock starts
                    w = cfg[s]["warmup"]
                    await asyncio.gather(*(x.call_tool(w["tool"], arguments=w.get("args", {})) for x in self.sessions[s]))
                for tool in (await self.sessions[s][0].list_tools()).tools:
                    self.tools[f"{slug(s)}__{tool.name}"] = (s, tool)
            self.ready.set()
            await self.stop.wait()

    def execute(self, name, args):
        """(text, error) of one tool call; thread-safe, and concurrent calls overlap on the session."""
        server, tool = self.tools[name]
        with self.lock:
            self.turn += 1
            session = self.sessions[server][self.turn % len(self.sessions[server])]  # round-robin over the pool
        fut = asyncio.run_coroutine_threadsafe(session.call_tool(tool.name, arguments=args), self.loop)
        try:
            r = fut.result(CALL_TIMEOUT)
        except Exception as e:
            return f"{type(e).__name__}: {e}", True
        return result_text(r), bool(getattr(r, "is_error", getattr(r, "isError", False)))

    def tool_specs(self):
        return {n: list(_schema(t).get("properties", {})) for n, (s, t) in self.tools.items()}

    def tool_docs(self):
        """Signature, description and per-argument type, allowed values, default and required flag of every tool."""
        return "\n\n".join(_doc(n, t) for n, (s, t) in self.tools.items())

    def close(self):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.keeper.result(30)
        self.loop.call_soon_threadsafe(self.loop.stop)
