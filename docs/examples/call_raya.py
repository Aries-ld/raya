"""Standalone Raya HTTP example. Python standard library only; no model or SDK installation."""

import argparse
import base64
import json
import math
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward an API key to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description="Call the Raya System One decision API")
    parser.add_argument("--request", type=Path, required=True, help="JSON request template")
    media = parser.add_mutually_exclusive_group()
    media.add_argument(
        "--image", type=Path, help="Replace the template's image with this local file"
    )
    media.add_argument(
        "--video", type=Path, help="Replace the template's video with this local file"
    )
    parser.add_argument("--timeout", type=float, default=150, help="HTTP timeout in seconds")
    args = parser.parse_args()
    base_url = os.environ.get("RAYA_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("RAYA_API_KEY", "")
    if not base_url or not api_key:
        parser.error("Set RAYA_BASE_URL and RAYA_API_KEY before calling the API")
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        parser.error("RAYA_BASE_URL must be a server root URL, without /v1 or credentials")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        payload = json.loads(args.request.read_text(encoding="utf-8-sig"))
        if args.image or args.video:
            kind = "image" if args.image else "video"
            path = (args.image or args.video).expanduser()
            if path.stat().st_size > 64 * 1024 * 1024:
                parser.error("This example client limits media to the service's default 64MiB")
            mime = mimetypes.guess_type(path.name)[0]
            if not mime or not mime.startswith(kind + "/"):
                parser.error("Use an image/video file with an appropriate extension")
            if not isinstance(payload.get("state"), dict):
                parser.error("Media examples require a state object")
            payload["state"]["media"] = [
                {
                    "type": kind,
                    "url": f"data:{mime};base64,"
                    + base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            ]
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if b"REPLACE_WITH_BASE64" in encoded:
            parser.error("Pass --image/--video, or replace the template's media URL")
    except (OSError, ValueError, AttributeError) as exc:
        parser.error(str(exc))
    request = urllib.request.Request(
        base_url + "/v1/systemone",
        data=encoded,
        method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        response = urllib.request.build_opener(NoRedirect).open(request, timeout=args.timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Connection or timeout error: {exc}", file=sys.stderr)
        return 1
    with response:
        status, headers, raw = response.status, response.headers, response.read()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        print(
            f"HTTP {status}: response was not valid JSON; check the gateway/service",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    metrics = {
        "http_status": status,
        "request_id": headers.get("x-request-id"),
        "client_rt_ms": elapsed_ms,
    }
    for key in ("forward", "processing", "queue"):
        value = headers.get(f"X-Raya-{key}-Ms")
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None
        metrics[key + "_ms"] = number if number is not None and math.isfinite(number) else None
    print(json.dumps(metrics, ensure_ascii=False), file=sys.stderr)
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
