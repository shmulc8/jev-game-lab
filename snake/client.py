"""Minimal HTTP client for the TypeSafe systemone API. Stdlib only."""

import json
import os
import time
import sys
import http.client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jevhttp

API_URL = "https://api.typesafe.ai/v1/systemone"

INPUT_TOKEN_PRICE_USD = 0.042 / 1_000_000

_api_key_cache = None


def api_key():
    global _api_key_cache
    if _api_key_cache is not None:
        return _api_key_cache
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY not found in environment")
    _api_key_cache = key
    return key


def ask(state, questions, model="jev-latest", max_attempts=3):
    payload = {"state": state, "model": model, "questions": questions}
    delay = 1.0
    last_error = None
    for attempt in range(max_attempts):
        start = time.perf_counter()
        try:
            status, raw = jevhttp.post_json(API_URL, payload, api_key(), timeout=30)
        except (OSError, http.client.HTTPException) as e:
            last_error = str(e)
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"network error: {last_error}")
        elapsed_ms = (time.perf_counter() - start) * 1000
        if status == 200:
            return json.loads(raw.decode("utf-8")), elapsed_ms
        error_body = raw.decode("utf-8", errors="replace")
        if status == 422:
            raise RuntimeError(f"422 validation error: {error_body}")
        if status in (429, 529) and attempt < max_attempts - 1:
            last_error = f"{status}: {error_body}"
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"{status} error: {error_body}")
    raise RuntimeError(f"exhausted retries: {last_error}")
