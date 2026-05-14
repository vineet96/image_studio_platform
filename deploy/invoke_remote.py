#!/usr/bin/env python3
"""
Call the deployed Image Studio root agent on Agent Engine.

Three modes:

  1. Smoke test — just send a text message, get streaming events.
       python deploy/invoke_remote.py --resource-id <ID> \
           --message "hello, are you deployed?"

  2. Multi-turn — reuse a session across calls.
       python deploy/invoke_remote.py --resource-id <ID> \
           --session-id <session_id_from_a_previous_run> \
           --message "now build the campaign"

  3. End-to-end with file uploads — attach images / PDF and trigger
     the full pipeline.
       python deploy/invoke_remote.py --resource-id <ID> \
           --attach person.jpg \
           --attach shirt.jpg \
           --attach brand.pdf \
           --message "try this shirt on the person, then build the campaign"

Output handling:
  - Text events stream to stdout as they arrive.
  - Image artifacts (data: URIs in agent replies AND artifact_delta
    events) are extracted and saved under --output-dir
    (default ./remote_outputs).
  - Use --raw to see the unfiltered event dicts (verbose; base64 will
    be elided to keep stdout readable).

Env defaults:
  GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
  AGENT_ENGINE_RESOURCE_ID.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any


# vertexai is lazy-imported in run() so --help works without the SDK
# installed (useful for CI doc-generation).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_URI_RE = re.compile(r"data:image/(\w+);base64,([A-Za-z0-9+/=]+)")


def _encode_attachment(path: Path) -> dict[str, Any]:
    """Build an inline_data part dict for an Agent Engine query.

    Agent Engine accepts the same Part format as the Gemini API: a dict
    with `inline_data: {mime_type, data}` where data is base64-encoded
    bytes.
    """
    if not path.is_file():
        raise SystemExit(f"Attachment not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


def _build_message(message: str, attachments: list[Path]) -> Any:
    """Return either a plain string (no attachments) or a list of Parts."""
    if not attachments:
        return message
    parts = [_encode_attachment(p) for p in attachments]
    parts.append({"text": message})
    return parts


def _extract_images_from_text(text: str, out_dir: Path, prefix: str) -> int:
    """Find data:image/...;base64,... URIs in `text`, save each to disk."""
    n = 0
    for m in DATA_URI_RE.finditer(text):
        ext, b64 = m.group(1), m.group(2)
        try:
            data = base64.b64decode(b64)
        except Exception:
            continue
        if len(data) < 100:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix}_{n:02d}.{ext}"
        (out_dir / fname).write_bytes(data)
        print(f"   ↳ saved {out_dir / fname} ({len(data):,} bytes)")
        n += 1
    return n


def _elide_base64(obj: Any) -> Any:
    """Recursively shorten base64-ish strings in event payloads so --raw
    output stays human-readable.
    """
    if isinstance(obj, str):
        if len(obj) > 400 and re.match(r"^[A-Za-z0-9+/=]+$", obj):
            return f"<base64 {len(obj)} chars elided>"
        if obj.startswith("data:image/") and len(obj) > 400:
            return f"<data URI {len(obj)} chars elided>"
        return obj
    if isinstance(obj, dict):
        return {k: _elide_base64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_elide_base64(v) for v in obj]
    return obj


def _print_event(event: Any, raw: bool, out_dir: Path) -> None:
    """Pretty-print one event from the streaming response."""
    if raw:
        print(json.dumps(_elide_base64(event), indent=2, default=str))
        return

    # Friendly view: pull out author + text parts; save any embedded images.
    author = event.get("author") if isinstance(event, dict) else None
    content = event.get("content") if isinstance(event, dict) else None
    if not content:
        return
    parts = content.get("parts", []) if isinstance(content, dict) else []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if "text" in part and part["text"]:
            text = part["text"]
            print(f"\n[{author or 'agent'}]")
            print(text)
            # Save any data: URIs embedded in this text.
            saved = _extract_images_from_text(
                text, out_dir,
                prefix=f"{author or 'agent'}_{uuid.uuid4().hex[:6]}",
            )
            if saved:
                print(f"   ({saved} image(s) extracted)")
        elif "function_call" in part:
            fc = part["function_call"]
            print(f"   → call: {fc.get('name')}({list((fc.get('args') or {}).keys())})")
        elif "function_response" in part:
            fr = part["function_response"]
            name = fr.get("name")
            resp = fr.get("response", {}) if isinstance(fr.get("response"), dict) else {}
            status = resp.get("status")
            # Surface any artifact filenames the tool produced so the
            # user knows what's available in the artifact pane / local
            # outputs dir.
            artifact_keys = [
                "artifact_filename", "image_artifact_filename",
                "caption_artifact_filename",
            ]
            files = [resp[k] for k in artifact_keys if resp.get(k)]
            extra = f"  artifacts: {files}" if files else ""
            print(f"   ← {name} returned status={status}{extra}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    import vertexai
    from vertexai import agent_engines

    vertexai.init(project=args.project, location=args.location)
    full_name = (
        f"projects/{args.project}/locations/{args.location}"
        f"/reasoningEngines/{args.resource_id}"
    )
    engine = agent_engines.get(full_name)

    if args.session_id:
        session_id = args.session_id
        print(f"Reusing session: {session_id}")
    else:
        session = await engine.async_create_session(user_id=args.user_id)
        session_id = session["id"]
        print(f"New session: {session_id}")
        print(f"  (pass --session-id {session_id} to continue this "
              f"conversation in a later call)")

    attachments = [Path(p) for p in args.attach]
    message = _build_message(args.message, attachments)

    if attachments:
        print(f"Attaching {len(attachments)} file(s):")
        for p in attachments:
            print(f"  - {p}")

    out_dir = Path(args.output_dir)
    print(f"\nSending message: {args.message[:120]}"
          f"{'...' if len(args.message) > 120 else ''}\n")

    async for event in engine.async_stream_query(
        user_id=args.user_id,
        session_id=session_id,
        message=message,
    ):
        _print_event(event, raw=args.raw, out_dir=out_dir)

    print(f"\nDone. Any extracted images are in {out_dir.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",
                   default=os.getenv("GOOGLE_CLOUD_PROJECT"),
                   help="GCP project (or $GOOGLE_CLOUD_PROJECT).")
    p.add_argument("--location",
                   default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    p.add_argument("--resource-id",
                   default=os.getenv("AGENT_ENGINE_RESOURCE_ID"),
                   help="Reasoning-engine ID printed by deploy.py "
                        "(or $AGENT_ENGINE_RESOURCE_ID).")
    p.add_argument("--user-id",
                   default=os.getenv("USER_ID", f"user-{uuid.uuid4().hex[:8]}"),
                   help="Stable per-end-user ID. Default is a random "
                        "ephemeral one.")
    p.add_argument("--session-id",
                   help="Existing session ID to continue a multi-turn "
                        "conversation. Omit to create a new session.")
    p.add_argument("--message", default="hello, are you deployed?",
                   help="The user message to send.")
    p.add_argument("--attach", action="append", default=[],
                   metavar="PATH",
                   help="Attach a file (image or PDF). Repeat for multiple.")
    p.add_argument("--output-dir", default="./remote_outputs",
                   help="Where to save extracted images. "
                        "Default: ./remote_outputs")
    p.add_argument("--raw", action="store_true",
                   help="Print full event dicts (base64 will be elided).")
    args = p.parse_args()

    if not args.project or not args.resource_id:
        raise SystemExit(
            "Set --project and --resource-id (or env vars "
            "GOOGLE_CLOUD_PROJECT / AGENT_ENGINE_RESOURCE_ID)."
        )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
