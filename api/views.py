import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from functools import lru_cache

import httpx
from adrf.decorators import api_view          # pip install adrf
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from rest_framework.response import Response

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Model Registry ───────────────────────────────────────────────────────────
# Ordered by: speed tier, reliability, context window.
# Add/remove freely — failover handles the rest.

MODELS = [
    # Fast & small — best for short replies, low latency
    {"id": "meta-llama/llama-3.2-3b-instruct:free",  "tier": "fast",   "ctx": 131072},
    {"id": "google/gemma-3-4b-it:free",               "tier": "fast",   "ctx": 131072},

    # Mid — good quality, acceptable speed
    {"id": "meta-llama/llama-3.3-70b-instruct:free",  "tier": "mid",    "ctx": 131072},
    {"id": "google/gemma-4-27b-it:free",              "tier": "mid",    "ctx": 131072},
    {"id": "mistralai/mistral-7b-instruct:free",      "tier": "mid",    "ctx": 32768},

    # Heavy — slow but highest quality, last resort
    {"id": "deepseek/deepseek-r1:free",               "tier": "heavy",  "ctx": 163840},
    {"id": "deepseek/deepseek-chat-v3-0324:free",     "tier": "heavy",  "ctx": 163840},
]

# ─── Simple In-Process Stats (resets on server restart) ───────────────────────
# Tracks per-model success/failure to reorder dynamically at runtime.
# Replace with Redis in production for persistence across workers.

_model_stats: dict[str, dict] = defaultdict(lambda: {
    "successes": 0,
    "failures":  0,
    "total_ms":  0.0,
})


def _record(model_id: str, success: bool, elapsed_ms: float = 0):
    s = _model_stats[model_id]
    if success:
        s["successes"] += 1
        s["total_ms"]  += elapsed_ms
    else:
        s["failures"] += 1


def _score(model: dict) -> float:
    """
    Lower score = try first.
    Formula: failure_rate * 100 + avg_latency_seconds
    New models start with score 0 (given benefit of the doubt).
    """
    s = _model_stats[model["id"]]
    total = s["successes"] + s["failures"]
    if total == 0:
        return 0.0
    failure_rate = s["failures"] / total
    avg_latency  = (s["total_ms"] / s["successes"] / 1000) if s["successes"] else 0
    return failure_rate * 100 + avg_latency


def sorted_models() -> list[dict]:
    return sorted(MODELS, key=_score)


# ─── Simple Response Cache ─────────────────────────────────────────────────────
# LRU in-memory cache keyed on (messages_hash).
# Free, zero-dependency, works for identical repeated questions.
# ~200 byte overhead per entry; 256 entries ≈ negligible RAM.

_CACHE_SIZE = 256

@lru_cache(maxsize=_CACHE_SIZE)
def _cached_response(cache_key: str) -> dict | None:
    # Populated externally via _store_in_cache; lru_cache used only for lookup.
    return None


_response_cache: dict[str, dict] = {}  # key → {response, model_used, ts}


def _cache_key(messages: list[dict]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _get_cache(key: str) -> dict | None:
    entry = _response_cache.get(key)
    if entry and (time.time() - entry["ts"]) < 300:   # 5-min TTL
        return entry
    return None


def _set_cache(key: str, data: dict):
    if len(_response_cache) >= _CACHE_SIZE:
        # Evict oldest entry
        oldest = min(_response_cache, key=lambda k: _response_cache[k]["ts"])
        del _response_cache[oldest]
    _response_cache[key] = {**data, "ts": time.time()}


# ─── Rate Limiter (per IP, in-memory) ─────────────────────────────────────────
# Free-tier APIs punish burst traffic. This keeps you safe.
# 10 requests / 60s per IP. Adjust freely.

_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT   = 10    # max requests
RATE_WINDOW  = 60    # seconds


def _is_rate_limited(ip: str) -> bool:
    now    = time.time()
    window = _rate_store[ip]
    _rate_store[ip] = [t for t in window if now - t < RATE_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        return True
    _rate_store[ip].append(now)
    return False


# ─── Core API Call ────────────────────────────────────────────────────────────

TOTAL_TIMEOUT  = 25.0   # hard wall-clock budget for the whole request
CONNECT_TIMEOUT = 3.0
MIN_READ_BUDGET = 2.0   # don't bother calling if less than this remains


async def _try_model(
    client:    httpx.AsyncClient,
    model_id:  str,
    messages:  list[dict],
    budget:    float,
) -> str | None:
    """
    Returns the assistant reply string, or None on any failure.
    """
    read_timeout = max(MIN_READ_BUDGET, min(10.0, budget - 1.0))
    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=read_timeout, write=2.0, pool=1.0)

    t0 = time.monotonic()
    try:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  os.getenv("SITE_URL", "http://localhost"),
                "X-Title":       os.getenv("APP_NAME",  "MyApp"),
            },
            json={
                "model":    model_id,
                "messages": messages,
                # Ask for concise replies on free tier to cut latency
                "max_tokens": int(os.getenv("MAX_TOKENS", 1024)),
            },
            timeout=timeout,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            body = {}
            try:    body = resp.json()
            except Exception: pass
            logger.warning("model=%s status=%d error=%s", model_id, resp.status_code, body.get("error", ""))
            _record(model_id, success=False)
            return None

        data = resp.json()

        if "error" in data:
            logger.warning("model=%s api_error=%s", model_id, data["error"])
            _record(model_id, success=False)
            return None

        content = data["choices"][0]["message"]["content"]
        _record(model_id, success=True, elapsed_ms=elapsed_ms)
        logger.info("model=%s ok latency=%.0fms", model_id, elapsed_ms)
        return content

    except httpx.TimeoutException:
        logger.warning("model=%s timed_out elapsed=%.1fs", model_id, time.monotonic() - t0)
        _record(model_id, success=False)
        return None
    except httpx.RequestError as exc:
        logger.warning("model=%s request_error=%s", model_id, exc)
        _record(model_id, success=False)
        return None
    except Exception as exc:
        logger.exception("model=%s unexpected error=%s", model_id, exc)
        _record(model_id, success=False)
        return None


# ─── Message Sanitiser ────────────────────────────────────────────────────────

ERROR_PREFIXES = (
    "Sorry, I couldn't",
    "All models are busy",
    "[error]",
)


def _sanitise_history(history: list[dict]) -> list[dict]:
    """
    Convert frontend history format → OpenAI messages format.
    Drops error messages, enforces alternating user/assistant turns.
    """
    messages: list[dict] = []
    for msg in history:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if any(content.startswith(p) for p in ERROR_PREFIXES):
            continue
        role = "assistant" if msg.get("role") in ("model", "assistant") else "user"
        # Collapse consecutive same-role messages (can confuse some models)
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})
    return messages


# ─── View ─────────────────────────────────────────────────────────────────────

@api_view(["POST"])
async def chat(request):
    # 1. Rate limiting
    ip = request.META.get("HTTP_X_FORWARDED_FOR", request.META.get("REMOTE_ADDR", "unknown"))
    ip = ip.split(",")[0].strip()
    if _is_rate_limited(ip):
        return Response({"error": "Too many requests. Slow down a bit."}, status=429)

    # 2. Parse & validate
    data    = request.data
    message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not message:
        return Response({"error": "Message cannot be empty."}, status=400)
    if len(message) > 8000:
        return Response({"error": "Message too long (max 8000 chars)."}, status=400)

    # 3. Build messages
    messages = _sanitise_history(history)
    messages.append({"role": "user", "content": message})

    # 4. Cache lookup
    key    = _cache_key(messages)
    cached = _get_cache(key)
    if cached:
        logger.info("cache_hit key=%s", key[:12])
        return Response({
            "response":   cached["response"],
            "model_used": cached["model_used"],
            "cached":     True,
        })

    # 5. Failover loop
    deadline = time.monotonic() + TOTAL_TIMEOUT

    async with httpx.AsyncClient() as client:
        for model in sorted_models():
            remaining = deadline - time.monotonic()
            if remaining < MIN_READ_BUDGET:
                logger.warning("budget_exhausted after %.1fs", TOTAL_TIMEOUT - remaining)
                break

            reply = await _try_model(client, model["id"], messages, budget=remaining)

            if reply is not None:
                _set_cache(key, {"response": reply, "model_used": model["id"]})
                return Response({
                    "response":   reply,
                    "model_used": model["id"],
                    "cached":     False,
                })

    return Response({"error": "All models are busy. Please try again in a moment."}, status=503)