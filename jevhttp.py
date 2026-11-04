"""Persistent HTTPS transport for the TypeSafe API.

One keep-alive connection per thread. A fresh TLS handshake to api.typesafe.ai costs
about 450 ms from here, so reusing the connection removes most of the fixed cost per call.
Stdlib only.
"""

import http.client
import json
import threading
import urllib.parse

_local = threading.local()


def _connection(host, timeout):
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "host", None) != host:
        if conn is not None:
            conn.close()
        conn = http.client.HTTPSConnection(host, timeout=timeout)
        _local.conn = conn
        _local.host = host
    conn.timeout = timeout
    if conn.sock is not None:
        conn.sock.settimeout(timeout)
    return conn


def _drop():
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
    _local.conn = None


def post_json(url, payload, api_key, timeout=60):
    """POST payload as JSON over a reused connection. Returns (status, raw_body_bytes).

    Raises OSError / http.client.HTTPException on network failure after one reconnect.
    """
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "api.typesafe.ai"
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment):
        raise ValueError("Only the official TypeSafe HTTPS API endpoint is supported")
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    for attempt in range(2):
        conn = _connection(parsed.netloc, timeout)
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.getheader("Connection", "").lower() == "close":
                _drop()
            return resp.status, raw
        except (http.client.HTTPException, OSError):
            _drop()
            if attempt == 1:
                raise
