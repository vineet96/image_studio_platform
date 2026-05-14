"""
Image Studio ADK agent — two image capabilities exposed as tools:

  * `virtual_try_on`  — Vertex AI `virtual-try-on-001`. Person + product
                        photo → person wearing product.
  * `edit_image`      — Vertex AI Nano Banana (Gemini image). Image +
                        natural-language instruction → edited image
                        (e.g. "change the shoes to red").

Uploads are auto-saved as session artifacts via SaveFilesAsArtifactsPlugin,
both tools save outputs as new artifacts, and the agent embeds the
generated images inline with markdown image syntax so the chat UI renders
them.

Run from the directory ABOVE this `vto_agent/` folder:
    adk web              # browser UI at http://localhost:8000
    adk run vto_agent    # CLI
"""

from __future__ import annotations

import io
import os
import re
import traceback
import uuid
from typing import Optional

# Wrapped imports so that any failure at import time prints the real
# traceback instead of being masked as "no root_agent found".
try:
    from google import genai
    from google.genai import types
    from google.genai.types import Image, ProductImage, RecontextImageSource

    from google.adk.agents import Agent
    from google.adk.apps import App
    from google.adk.plugins.save_files_as_artifacts_plugin import (
        SaveFilesAsArtifactsPlugin,
    )
    from google.adk.tools import ToolContext
except Exception:
    traceback.print_exc()
    raise


# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
# Override via env var if your project has access to a preview model.
#   gemini-2.5-flash-image           — GA (default)
#   gemini-3.1-flash-image-preview   — Nano Banana 2 (allowlist)
#   gemini-3-pro-image-preview       — Nano Banana Pro (allowlist)
NANO_BANANA_MODEL = os.getenv("NANO_BANANA_MODEL", "gemini-2.5-flash-image")
VTO_MODEL = "virtual-try-on-001"


# ---------------------------------------------------------------------------
# Vertex Gen AI client
# ---------------------------------------------------------------------------
# Required env (loaded from vto_agent/.env automatically by `adk web`;
# set automatically by deploy.py for Agent Engine):
#   GOOGLE_CLOUD_PROJECT
#   GOOGLE_CLOUD_LOCATION  (region — NOT "global")
#   GOOGLE_GENAI_USE_VERTEXAI=True
#
# Force vertexai=True explicitly so the client never falls back to the
# Gemini Developer API key path. Otherwise, if GOOGLE_GENAI_USE_VERTEXAI
# is unset (e.g. when running deploy.py from a fresh shell), the client
# constructor errors with "No API key was provided" at import time.
_VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
_VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
_client = genai.Client(
    vertexai=True,
    project=_VERTEX_PROJECT or None,
    location=_VERTEX_LOCATION or None,
)


# ===========================================================================
# Artifact resolution helpers
# ===========================================================================

async def _resolve_image_part(
    name_or_auto: str,
    tool_context: ToolContext,
    *,
    exclude_results: bool = False,
) -> Optional[types.Part]:
    """Resolve a string from the LLM into an image artifact Part.

    Accepts:
      - "auto" → most relevant image in the session, in this priority:
            1. state['latest_hero_image'] — the cursor set by
               virtual_try_on and edit_image each time they succeed.
               This makes "edit the VTO output" and "edit the previous
               edit" work naturally — the LLM doesn't have to pass a
               filename and the runtime always picks the right image.
            2. Newest `edit_result_*` artifact.
            3. Newest `vto_result_*` artifact.
            4. Newest non-result image artifact (a raw user upload).
      - explicit artifact filename → loaded directly.
      - gs:// URI → fetched (not implemented here; left as TODO).

    `exclude_results=True` skips steps 2 and 3 above (useful only when
    callers explicitly need to find a user upload and not a tool
    output). Default is False because the common case — "edit my VTO
    result" — needs the tool output to win.

    Returns None if nothing matches.
    """
    if name_or_auto.startswith("gs://"):
        raise NotImplementedError(
            "gs:// URIs are not handled by this dev agent. "
            "Upload the file via the chat UI instead."
        )

    if name_or_auto != "auto":
        return await tool_context.load_artifact(filename=name_or_auto)

    # 1. Trust the state cursor first.
    cursor = tool_context.state.get("latest_hero_image")
    if cursor:
        try:
            part = await tool_context.load_artifact(filename=cursor)
        except Exception:
            part = None
        if (part and part.inline_data
                and part.inline_data.mime_type
                and part.inline_data.mime_type.startswith("image/")):
            return part
        # Cursor stale → fall through to scan.

    names = await tool_context.list_artifacts()

    def _is_image(p) -> bool:
        return (p is not None
                and p.inline_data is not None
                and p.inline_data.mime_type is not None
                and p.inline_data.mime_type.startswith("image/"))

    # 2. Newest edit_result_*.
    if not exclude_results:
        for n in reversed(names):
            if not n.startswith("edit_result_"):
                continue
            part = await tool_context.load_artifact(filename=n)
            if _is_image(part):
                return part

    # 3. Newest vto_result_*.
    if not exclude_results:
        for n in reversed(names):
            if not n.startswith("vto_result_"):
                continue
            part = await tool_context.load_artifact(filename=n)
            if _is_image(part):
                return part

    # 4. Newest raw upload.
    for n in reversed(names):
        if n.startswith("vto_result_") or n.startswith("edit_result_"):
            continue
        part = await tool_context.load_artifact(filename=n)
        if _is_image(part):
            return part

    return None


def _format_model_error(e: Exception, model_id: str) -> str:
    """Make access-denied errors actionable instead of cryptic."""
    msg = str(e)
    if "was not found" in msg or "PermissionDenied" in type(e).__name__:
        return (
            f"Your GCP project doesn't have access to '{model_id}'. "
            f"This usually means the model is allowlist-only (preview). "
            f"Either request access on the Google AI Developers Forum, or "
            f"switch to a GA model (e.g. set "
            f"NANO_BANANA_MODEL=gemini-2.5-flash-image in your .env)."
        )
    return f"{type(e).__name__}: {e}"


# ===========================================================================
# Tool 0: Reset session
# ===========================================================================

# State keys we clear on reset. Anything else in state (ADK internals,
# user-set memory, etc.) is left alone.
_RESET_STATE_KEYS = (
    "latest_hero_image",
    "vto_image_filename",
    "brand_pdf_filename",
    "brand_pdf_source",
    "brand",
    "brand_summary",
    "creative_plan",
    "creative_plan_ack",
    "_transferred_at_user_turn",
    "_social_pending",
    "_brand_pdf_gcs_error",
)


async def reset_session(tool_context: ToolContext) -> dict:
    """Clear carry-over campaign state.

    When the user reuses the same chat session for a new campaign — a
    new person photo, a new product, a new brand — old state can
    silently leak into the new run (e.g. the campaign picks up the
    previous VTO as its hero). Calling this tool wipes those state
    keys without deleting artifacts.

    NOTE: ADK's State object has no .pop() and no __delitem__ — the
    only mutation primitives are __setitem__, update, and setdefault.
    So we "clear" by assigning None. Downstream code uses
    `state.get(key)` and `if value:` checks, which treat None and
    missing identically.

    Use when the user says "start over", "reset", "new campaign",
    "forget that", or similar.

    Returns:
        dict with keys: status, cleared_keys, message.
    """
    cleared = []
    for key in _RESET_STATE_KEYS:
        if key in tool_context.state and tool_context.state.get(key) is not None:
            tool_context.state[key] = None
            cleared.append(key)
    return {
        "status": "success",
        "cleared_keys": cleared,
        "message": (
            f"Session campaign state cleared ({len(cleared)} keys). "
            f"Upload fresh images and re-run virtual_try_on to start "
            f"a new campaign."
        ),
    }


# ===========================================================================
# Tool 1: Virtual try-on
# ===========================================================================

async def virtual_try_on(
    person_image: str,
    product_image: str,
    tool_context: ToolContext,
) -> dict:
    """Run Google's virtual try-on model to show a person wearing a product.

    Args:
        person_image: Artifact filename of the person photo, or "auto" to
            use the most-recent non-result image upload.
        product_image: Artifact filename of the clothing/product photo, or
            "auto" to use the second-most-recent upload.

    Returns:
        dict with keys: status, artifact_filename, markdown, message.
    """
    try:
        person_part = await _resolve_image_part(person_image, tool_context)
        if person_part is None:
            return {"status": "error",
                    "message": "Couldn't find a person image. Upload one."}

        # If both args are "auto", we still need two distinct images.
        product_part = await _resolve_image_part(product_image, tool_context)
        if product_part is None or product_part is person_part:
            return {"status": "error",
                    "message": "Couldn't find a product image. Upload one."}

        person_img = Image(
            image_bytes=person_part.inline_data.data,
            mime_type=person_part.inline_data.mime_type or "image/png",
        )
        product_img = Image(
            image_bytes=product_part.inline_data.data,
            mime_type=product_part.inline_data.mime_type or "image/png",
        )

        response = _client.models.recontext_image(
            model=VTO_MODEL,
            source=RecontextImageSource(
                person_image=person_img,
                product_images=[ProductImage(product_image=product_img)],
            ),
        )
        result_bytes = response.generated_images[0].image.image_bytes

        out_name = f"vto_result_{uuid.uuid4().hex[:8]}.png"
        await tool_context.save_artifact(
            filename=out_name,
            artifact=types.Part.from_bytes(
                data=result_bytes, mime_type="image/png"
            ),
        )

        # Update the cursor that downstream tools (campaign, social) use
        # to find "the latest image". Without this, find_vto_image would
        # only ever return the FIRST try-on result, not the most recent
        # one, and an `edit_image` follow-up would be ignored when
        # building ads / social posts. See find_vto_image for details.
        tool_context.state["latest_hero_image"] = out_name

        # Return only the artifact filename in the markdown reference.
        # Embedding base64 in tool returns blows the LLM context (every
        # function_response is replayed to the model on the next turn,
        # and 1MB of base64 ≈ 250K tokens). ADK Web renders artifacts by
        # filename via its artifact-fetch route; if your client doesn't,
        # tell the user to look at the artifacts pane.
        return {
            "status": "success",
            "artifact_filename": out_name,
            "markdown": f"![Try-on result]({out_name})",
        }

    except Exception as e:
        return {"status": "error",
                "message": _format_model_error(e, VTO_MODEL)}


# ===========================================================================
# Tool 2: Nano Banana image editing
# ===========================================================================

async def edit_image(
    edit_instruction: str,
    source_filename: str,
    tool_context: ToolContext,
) -> dict:
    """Apply a generative edit to a single uploaded image using Nano Banana.

    Examples of `edit_instruction`:
        "change the shoes to navy blue"
        "make the background a sunny beach"
        "remove the watermark"

    Args:
        edit_instruction: Specific natural-language edit. If the user
            was vague ("change the color"), the AGENT should ask a
            clarifying question first instead of calling this tool.
        source_filename: Either the artifact filename of the image to
            edit, or "auto" to let the tool pick the most relevant one.
            "auto" resolution order:
              1. state['latest_hero_image'] (set by virtual_try_on and
                 prior edit_image calls — this is what makes "edit the
                 VTO result" work without naming a file).
              2. Newest edit_result_* artifact.
              3. Newest vto_result_* artifact.
              4. Newest user-uploaded image.
            The LLM should pass "auto" by default. Only pass an explicit
            filename when the user has named a specific image.

    Returns:
        dict with keys: status, artifact_filename, markdown, message.
    """
    try:
        src_part = await _resolve_image_part(source_filename, tool_context)
        if src_part is None:
            return {
                "status": "error",
                "message": "No image found in the session to edit. "
                           "Either upload an image and try again, or run "
                           "virtual_try_on first so there's a try-on "
                           "result to edit."
            }

        # Detect single-attribute edits so we can tighten the prompt
        # past Nano Banana's tendency to treat "change X" as a license
        # to regenerate. Recolor / "make it Y" patterns are by far the
        # most common, and the most often mis-interpreted (the model
        # swaps the garment for a different type in the new color).
        EDIT_LOWER = edit_instruction.lower().strip()
        is_recolor = bool(re.search(
            r"\b(change|make|recolor|turn)\b.*\b(color|colour|"
            r"red|orange|yellow|green|blue|navy|teal|purple|pink|"
            r"black|white|gray|grey|brown|tan|beige|cream|gold|"
            r"silver|maroon|burgundy|olive|charcoal)\b",
            EDIT_LOWER,
        ))

        if is_recolor:
            # Recolor-only path: the strongest signal we can give the
            # model is "this is a pixel-level color change, do nothing
            # else". Naming the operation explicitly cuts down on the
            # garment-swap failure mode dramatically.
            wrapped_prompt = (
                "TASK TYPE: pixel-level recolor. Take the attached "
                "photograph and change ONLY the color specified below. "
                "Do not regenerate, restyle, or modify anything else.\n\n"
                f"COLOR CHANGE REQUESTED: {edit_instruction}\n\n"
                "HARD CONSTRAINTS (treat as a forbidden list):\n"
                "- Do NOT change the garment TYPE (a coat stays a coat, "
                "a shirt stays a shirt, a dress stays a dress).\n"
                "- Do NOT change the garment CUT, SILHOUETTE, LENGTH, "
                "COLLAR, LAPELS, SLEEVES, BUTTONS, ZIPPERS, POCKETS, "
                "SEAMS, or FABRIC TEXTURE.\n"
                "- Do NOT change the person's face, identity, hair, "
                "skin tone, body shape, or pose.\n"
                "- Do NOT change the background, lighting, framing, or "
                "camera angle.\n"
                "- Do NOT change any other clothing item or accessory.\n\n"
                "WHAT TO DO: identify the colored region(s) the user "
                "wants changed, and replace ONLY their pixel hues with "
                "the requested color, preserving the original fabric "
                "shading, highlights, shadows, and material texture."
            )
        else:
            # General-purpose edit: structured prompt with explicit
            # preservation rules. Same shape as before — works well for
            # background swaps, object removal, etc.
            wrapped_prompt = (
                "Edit the attached photograph while preserving "
                "EVERYTHING that is not explicitly being changed.\n\n"
                f"EDIT TO APPLY: {edit_instruction}\n\n"
                "PRESERVE EXACTLY (do NOT alter these):\n"
                "- The person's face, identity, hair, skin tone, body "
                "shape, and pose.\n"
                "- The garment's overall TYPE, CUT, SILHOUETTE, FABRIC, "
                "and FIT. If the edit is about color, change ONLY the "
                "color — keep the exact same garment design, collar "
                "style, sleeve length, hem, buttons, seams, and texture.\n"
                "- All other clothing items and accessories not "
                "mentioned in the edit.\n"
                "- The background, lighting, framing, composition, and "
                "camera angle, unless the edit explicitly changes one "
                "of those.\n"
                "- Image resolution and aspect ratio.\n\n"
                "OUTPUT: a photograph that looks like the original "
                "with ONLY the requested change applied. Do not "
                "regenerate the scene from scratch — modify the source "
                "pixels."
            )

        response = _client.models.generate_content(
            model=NANO_BANANA_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=src_part.inline_data.data,
                    mime_type=src_part.inline_data.mime_type or "image/png",
                ),
                wrapped_prompt,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        image_bytes = None
        for cand in response.candidates or []:
            for part in cand.content.parts or []:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    break
            if image_bytes:
                break
        if image_bytes is None:
            return {"status": "error",
                    "message": "Model returned no image. Try rephrasing."}

        out_name = f"edit_result_{uuid.uuid4().hex[:8]}.png"
        await tool_context.save_artifact(
            filename=out_name,
            artifact=types.Part.from_bytes(
                data=image_bytes, mime_type="image/png"
            ),
        )

        # Update the cursor used by the campaign / social pipelines.
        # This is what makes "edit the VTO result, then build ads" use
        # the edited image instead of the original VTO output.
        tool_context.state["latest_hero_image"] = out_name

        # Filename-only reference; see note in virtual_try_on.
        return {
            "status": "success",
            "artifact_filename": out_name,
            "markdown": f"![Edited image]({out_name})",
        }

    except Exception as e:
        return {"status": "error",
                "message": _format_model_error(e, NANO_BANANA_MODEL)}


# ===========================================================================
# Agent + App
# ===========================================================================

INSTRUCTION = """\
You help users with image generation tasks. You have two tools:

A) VIRTUAL TRY-ON — show a person wearing a clothing product.
   Use the `virtual_try_on` tool. Requires TWO uploaded images: a person
   photo and a clothing/product photo. If either is missing, ask the user
   to upload both. Pass the filenames you can see in the conversation, or
   "auto" for both if you can't tell which is which.

B) IMAGE EDITING (Nano Banana) — apply a generative edit to a single
   uploaded image. Use the `edit_image` tool. Examples: change a color,
   swap an outfit, remove an object, change the background.

   IMPORTANT: Before calling `edit_image`, make sure you have BOTH:
     1. A clear, SPECIFIC instruction (e.g. "change shoes to red", not
        just "change the color"). If the user is vague, ASK clarifying
        questions first — in particular, if they say "change the color"
        without naming one, ask "Which color would you like?" and wait
        for their reply.
     2. An uploaded image to edit. If none is uploaded, ask for one.

   Once you have both, call `edit_image(edit_instruction, source_filename)`.
   DEFAULT: pass "auto" for source_filename. The tool will then pick the
   most recent image automatically (the VTO output, the last edit, or
   the user's upload). Only pass an explicit filename when the user
   has named a specific image to edit.

For BOTH tools: when they succeed, reply with a SHORT confirmation and
embed the generated image using the `markdown` field returned by the
tool, which looks like:  ![Edited image](edit_result_xxxxxxxx.png)
ADK Web renders saved image artifacts inline when you reference them
this way.

If a tool returns an error, explain it plainly and suggest a fix.
Keep replies short. Never echo base64 or raw bytes.
"""

try:
    root_agent = Agent(
        name="image_studio_agent",
        model="gemini-2.5-flash",
        description=(
            "Image studio: virtual try-on (person + clothing) and "
            "generative image editing using Google's virtual-try-on-001 "
            "and Nano Banana models."
        ),
        instruction=INSTRUCTION,
        tools=[virtual_try_on, edit_image, reset_session],
    )

    # `App.name` must equal the package directory name (`vto_agent`) so
    # the ADK CLI's session keying matches what runtime calls use. See
    # image_studio_root_agent/agent.py for the full rationale.
    app = App(
        name="vto_agent",
        root_agent=root_agent,
        plugins=[SaveFilesAsArtifactsPlugin()],
    )
except Exception:
    traceback.print_exc()
    raise