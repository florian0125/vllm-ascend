#!/usr/bin/env python3
"""
Download a model from ModelScope with SSL certificate verification disabled.

Takes either a full model URL or a bare <owner>/<name> id.

Requires:  pip install "modelscope>=1.18" requests urllib3

Examples:
    python download_modelscope.py
    python download_modelscope.py https://www.modelscope.cn/models/gdydems/DeepSeek-V4-Flash-w4a8-mtp
    python download_modelscope.py --url https://www.modelscope.cn/models/Qwen/Qwen3-8B/files
    python download_modelscope.py Qwen/Qwen3-8B --cache-dir /data/models --workers 8
    python download_modelscope.py --exclude "*.bin" --include "*.safetensors"
"""

from __future__ import annotations

import argparse
import inspect
import os
import ssl
import sys
import time
from urllib.parse import parse_qs, urlparse

DEFAULT_MODEL = "https://www.modelscope.cn/models/gdydems/DeepSeek-V4-Flash-w4a8-mtp"


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    """Turn a ModelScope model URL (or a bare id) into ('owner/name', revision).

    Accepts, among others:
        https://www.modelscope.cn/models/gdydems/DeepSeek-V4-Flash-w4a8-mtp
        https://modelscope.cn/models/Qwen/Qwen3-8B/files?Revision=master
        https://www.modelscope.cn/zh-CN/models/Qwen/Qwen3-8B/summary
        Qwen/Qwen3-8B
    """
    ref = ref.strip().strip("'\"")
    revision = None

    if "://" in ref or ref.startswith("www."):
        parsed = urlparse(ref if "://" in ref else f"https://{ref}")
        if "modelscope" not in parsed.netloc:
            raise ValueError(f"not a modelscope.cn URL: {ref}")
        parts = [p for p in parsed.path.split("/") if p]
        if "models" in parts:                       # drop any locale prefix too
            parts = parts[parts.index("models") + 1:]
        elif "model" in parts:
            parts = parts[parts.index("model") + 1:]
        revision = parse_qs(parsed.query).get("Revision", [None])[0]
    else:
        parts = [p for p in ref.split("/") if p]

    # Anything past owner/name is a UI route (files, summary, tree/<rev>, ...)
    if len(parts) >= 3 and parts[2] in {"tree", "blob"} and len(parts) >= 4:
        revision = revision or parts[3]
    if len(parts) < 2:
        raise ValueError(f"could not read an <owner>/<name> model id out of: {ref}")

    return f"{parts[0]}/{parts[1]}", revision


def disable_ssl_verification() -> None:
    """Make every HTTPS call in this process skip certificate checks.

    Call this BEFORE importing modelscope: the SDK builds its own
    requests.Session objects, so the patch has to be in place first.
    """
    # urllib / ssl-based calls (some code paths don't go through requests)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Stop requests from pulling a CA bundle out of the environment
    for var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        os.environ.pop(var, None)
    os.environ["CURL_CA_BUNDLE"] = ""

    import requests
    import urllib3
    from urllib3.exceptions import InsecureRequestWarning

    urllib3.disable_warnings(InsecureRequestWarning)

    # 1) Override whatever `verify` the caller passed in explicitly.
    _orig_request = requests.Session.request

    def request(self, *args, **kwargs):
        kwargs["verify"] = False
        return _orig_request(self, *args, **kwargs)

    requests.Session.request = request

    # 2) Belt and braces: also force it at the env-merge stage, which is
    #    what actually decides `verify` for sessions built elsewhere.
    _orig_merge = requests.Session.merge_environment_settings

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = _orig_merge(self, url, proxies, stream, verify, cert)
        settings["verify"] = False
        return settings

    requests.Session.merge_environment_settings = merge_environment_settings


def download(args: argparse.Namespace) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    kwargs = {
        "model_id": args.model_id,
        "cache_dir": args.cache_dir,
        "local_dir": args.local_dir,
        "revision": args.revision,
        "allow_patterns": args.include or None,
        "ignore_patterns": args.exclude or None,
        "max_workers": args.workers,
    }

    # Older/newer SDK versions differ on which kwargs exist — keep only the
    # ones this installed version actually accepts.
    supported = inspect.signature(snapshot_download).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in supported and v is not None}

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            print(f"[attempt {attempt}/{args.retries}] downloading {args.model_id} ...")
            return snapshot_download(**kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # network flakiness on multi-GB repos is normal
            last_error = exc
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if attempt < args.retries:
                wait = min(60, 5 * attempt)
                print(f"  retrying in {wait}s (partial files are resumed) ...")
                time.sleep(wait)

    raise SystemExit(f"Giving up after {args.retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default=None,
                   help="Model URL (https://www.modelscope.cn/models/<owner>/<name>) "
                        "or bare <owner>/<name> id. "
                        f"Default: {DEFAULT_MODEL}")
    p.add_argument("-u", "--url", "--model-id", dest="url", default=None,
                   help="Same as the positional argument; accepts a URL or an id")
    p.add_argument("--cache-dir", default="./models",
                   help="ModelScope cache root (default: ./models)")
    p.add_argument("--local-dir", default=None,
                   help="Download flat into this dir instead of the cache layout")
    p.add_argument("--revision", default=None, help="Branch/tag/commit (default: master)")
    p.add_argument("--include", nargs="*", default=None,
                   help="Glob patterns to download, e.g. '*.safetensors' '*.json'")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Glob patterns to skip, e.g. '*.bin'")
    p.add_argument("--workers", type=int, default=4, help="Parallel file downloads")
    p.add_argument("--retries", type=int, default=5, help="Retry attempts on failure")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.url and args.model and args.url != args.model:
        raise SystemExit("Give the model once: either positionally or via --url, not both.")

    try:
        args.model_id, url_revision = parse_model_ref(args.url or args.model or DEFAULT_MODEL)
    except ValueError as exc:
        raise SystemExit(f"Bad model argument — {exc}")
    args.revision = args.revision or url_revision

    print(f"Model:    {args.model_id}" + (f"  (revision: {args.revision})" if args.revision else ""))

    disable_ssl_verification()
    print("SSL certificate verification is DISABLED for this process.")
    path = download(args)
    print(f"\nDone. Model files are in:\n  {os.path.abspath(path)}")


if __name__ == "__main__":
    main()