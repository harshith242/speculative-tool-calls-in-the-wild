"""OpenAI-compatible chat client (DeepSeek or Ollama) with a disk cache, retry/backoff, usage and spend reporting.
A reply cached by an earlier run counts toward usage (logical cost) once per process, never toward real spend.
The cache key has no model digest: clear cache/llm after changing an Ollama Modelfile."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import openai
from tqdm import tqdm

from fastlane.files import write_atomic

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
MAX_WAIT = 120  # a longer retry-after means a daily limit: stop and resume later instead of sleeping
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "provider_cached_tokens", "latency_s")  # summed per call


class ProviderExhausted(Exception):
    pass


class ReplayMiss(Exception):
    pass


def usd(usage, prices):
    """Dollar cost of a usage record; prices are USD per 1M tokens (missing prices = free, e.g. local)."""
    if not prices:
        return 0.0
    hit = usage.get("provider_cached_tokens", 0)
    miss = usage.get("prompt_tokens", 0) - hit
    output = usage.get("completion_tokens", 0)
    return (hit * prices["cache_hit"] + miss * prices["cache_miss"] + output * prices["output"]) / 1e6


def extract_json(text):
    """The first {...} object in a reply, or None."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def _cache_hit_tokens(usage):
    """Prompt tokens served from the provider's prefix cache (DeepSeek or OpenAI-style field)."""
    hits = getattr(usage, "prompt_cache_hit_tokens", None)
    if hits is None:
        details = getattr(usage, "prompt_tokens_details", None)
        hits = getattr(details, "cached_tokens", 0) if details else 0
    return hits or 0


class LLM:
    def __init__(self, model, base_url=None, api_key=None, cache_dir="cache/llm", client=None, max_tries=10,
                 options=None, prices=None, on_spend=None, replay_only=False):
        self.model = model
        self.client = client or openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=600)
        self.cache_dir = Path(cache_dir) if cache_dir else None  # None: live calls, never cached (timing must be real)
        self.max_tries = max_tries
        self.options = options or {}  # extra request params, e.g. {"reasoning_effort": "low"}
        self.prices, self.on_spend = prices, on_spend  # on_spend(usd) runs after every real (non-cached) call
        self.replay_only = replay_only  # a cache miss raises ReplayMiss instead of calling the API
        self.usage = {"calls": 0, **dict.fromkeys(USAGE_KEYS, 0)}
        self.seen = set()  # cache keys already counted in this process: in-run replays are free

    def chat(self, messages, tools=None):
        """{content, reasoning, tool_calls: [{id, name, arguments}], model, USAGE_KEYS}; replays keep latency_s."""
        if self.cache_dir is None:
            reply = self._call(messages, tools)
            self._record(reply)
            return reply
        payload = [self.model, self.options, messages, tools, 0.0, 0]  # temperature, sample: kept for the cache keys
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            reply = json.loads(path.read_text())
            if key in self.seen:
                return reply
        else:
            if self.replay_only:
                raise ReplayMiss(f"{self.model}: an old arm needed a new call (cache miss); nothing was spent")
            reply = self._call(messages, tools)
            write_atomic(path, json.dumps(reply))
            if self.on_spend:
                self.on_spend(usd(reply, self.prices))
        self.seen.add(key)
        self.usage["calls"] += 1
        for k in USAGE_KEYS:
            self.usage[k] += reply.get(k, 0)
        return reply

    def _record(self, reply):
        """Count one real (non-cached) call's usage and spend."""
        if self.on_spend:
            self.on_spend(usd(reply, self.prices))
        self.usage["calls"] += 1
        for k in USAGE_KEYS:
            self.usage[k] += reply.get(k, 0)

    def stream(self, messages, on_text=None):
        """Streamed, never cached: {content, chunks: [[t, text]], t_first, latency_s, retried, usage}; on_text gets the text so far."""
        retried, last = False, None
        for attempt in range(self.max_tries):
            start, chunks, usage, text = time.monotonic(), [], None, ""
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.0,
                                                           stream=True, stream_options={"include_usage": True},
                                                           **self.options)
                for event in resp:
                    usage = event.usage or usage
                    if event.choices and event.choices[0].delta.content:
                        piece = event.choices[0].delta.content
                        chunks.append([round(time.monotonic() - start, 3), piece])
                        text += piece
                        if on_text:
                            on_text(text)
            except (openai.RateLimitError, *RETRYABLE) as e:
                last, retried = e, True
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            reply = {"content": text, "chunks": chunks, "t_first": chunks[0][0] if chunks else None,
                     "latency_s": round(time.monotonic() - start, 3), "retried": retried,
                     "prompt_tokens": usage.prompt_tokens if usage else 0,
                     "completion_tokens": usage.completion_tokens if usage else 0,
                     "provider_cached_tokens": _cache_hit_tokens(usage) if usage else 0}
            self._record(reply)
            return reply
        raise ProviderExhausted(f"{self.model}: {last}")

    def _call(self, messages, tools):
        extra = {"tools": tools, **self.options} if tools else dict(self.options)
        for attempt in range(self.max_tries):
            backoff = min(2 ** (attempt + 1), 60)
            start = time.monotonic()
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.0,
                                                           **extra)
            except openai.RateLimitError as e:
                last = e
                wait = float(e.response.headers.get("retry-after") or backoff)
                if wait > MAX_WAIT:
                    raise ProviderExhausted(f"{self.model}: rate limited for {wait:.0f}s (daily limit?)")
                tqdm.write(f"  {self.model}: rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            except RETRYABLE as e:
                last = e
                time.sleep(backoff)
                continue
            if not getattr(resp, "choices", None):
                # Some providers answer 200 with an error body and no choices when the upstream model fails.
                last = f"empty response: {getattr(resp, 'error', None)}"
                time.sleep(backoff)
                continue
            msg = resp.choices[0].message
            calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                     for c in msg.tool_calls or []]
            return {
                "content": msg.content,
                "reasoning": getattr(msg, "reasoning_content", None),  # DeepSeek thinking; must be sent back with tools
                "tool_calls": calls,
                "model": resp.model,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "provider_cached_tokens": _cache_hit_tokens(resp.usage),
                "latency_s": round(time.monotonic() - start, 3),
            }
        raise ProviderExhausted(f"{self.model}: {last}")


def make_llm(role_cfg, cache_dir, on_spend=None, replay_only=False):
    """LLM for a config role (agent profile or proposer); a role without api_key_env is a local server."""
    key_env = role_cfg.get("api_key_env")
    api_key = os.environ[key_env] if key_env else "local"
    return LLM(role_cfg["model"], role_cfg["base_url"], api_key, cache_dir, options=role_cfg.get("options"),
               prices=role_cfg.get("usd_per_million"), on_spend=on_spend, replay_only=replay_only)
