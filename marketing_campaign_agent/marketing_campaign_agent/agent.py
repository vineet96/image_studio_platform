"""
Marketing Campaign ADK Agent
============================

A sub-agent for the Virtual Try-On agent. Given:
  - the VTO output image (person wearing product), and
  - a brand guidelines PDF,

it produces a Google Ads **Responsive Display** asset set:
  - Landscape 1.91:1   (1200 x 628)  — required
  - Square    1:1      (1200 x 1200) — required
  - Logo      4:1      (1200 x 300)  — required
  - Logo Sq.  1:1      (1200 x 1200) — required

Spec source: Google Ads "Image requirements" for Display / RDA.
File format: PNG/JPG, <5MB each.

Architecture (SequentialAgent pipeline):
    brand_parser_agent      -> reads brand PDF, extracts palette/voice/rules
    creative_director_agent -> plans the 4 creatives (prompts + copy)
    asset_generator_agent   -> calls Nano Banana / Imagen 4 tools to render

The whole pipeline is exposed as `root_agent` so it can run standalone via
`adk web` / `adk run marketing_campaign_agent`, AND it's exported as
`marketing_campaign_agent` so the VTO agent can mount it under `sub_agents=[...]`.
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types
from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import ToolContext


# ---------------------------------------------------------------------------
# Vertex AI client (shared by all tools)
# ---------------------------------------------------------------------------
# Required env (set automatically by deploy.py for Agent Engine; set in
# .env for local `adk web`):
#   GOOGLE_CLOUD_PROJECT
#   GOOGLE_CLOUD_LOCATION   (must be a region, e.g. us-central1)
#   GOOGLE_GENAI_USE_VERTEXAI=True   (forced True below regardless)
#
# We construct the client with explicit vertexai=True so it never falls
# back to the Gemini Developer API key path. Without this, a missing or
# unset GOOGLE_GENAI_USE_VERTEXAI causes genai.Client() to look for an
# API key and raise "No API key was provided" — which has bitten us at
# deploy time when the local shell didn't have the env var set.
_VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
_VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
_client = genai.Client(
    vertexai=True,
    project=_VERTEX_PROJECT or None,
    location=_VERTEX_LOCATION or None,
)

# Local fallback directory for generated PNGs. Artifacts are preferred; this
# is only used if no ToolContext / artifact service is available.
_OUTPUT_DIR = Path(os.getenv("CAMPAIGN_OUTPUT_DIR", "./campaign_outputs"))
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Tools
# ===========================================================================

async def extract_brand_guidelines(
    pdf_artifact_filename: str,
    tool_context: ToolContext,
) -> dict:
    """Extract brand guidelines from a PDF using Gemini multimodal understanding.

    The user uploads a brand guidelines PDF; ADK saves it as an artifact via
    SaveFilesAsArtifactsPlugin. This tool loads those bytes, sends them to
    Gemini 2.5 Flash, and asks for a structured JSON summary of the brand:
    colors, typography, voice/tone, logo usage rules, dos/donts.

    Args:
        pdf_artifact_filename: Name of the artifact holding the brand PDF
            (e.g. "artifact_<invocation_id>_0").

    Returns:
        dict with keys:
          status: "success" | "error"
          brand: structured brand dict (when success)
          message: error detail (when error)
    """
    try:
        part = await tool_context.load_artifact(filename=pdf_artifact_filename)
        if part is None or part.inline_data is None:
            return {
                "status": "error",
                "message": f"Artifact '{pdf_artifact_filename}' not found.",
            }

        pdf_bytes = part.inline_data.data
        mime = part.inline_data.mime_type or "application/pdf"

        extraction_prompt = """\
You are reading a brand guidelines PDF. Extract the brand information that
is needed to design Google Display ads. Respond with ONLY a JSON object
(no markdown fences, no commentary). Use this exact schema:

{
  "brand_name": "string",
  "tagline": "string or null",
  "primary_colors": [{"name": "...", "hex": "#RRGGBB"}],
  "secondary_colors": [{"name": "...", "hex": "#RRGGBB"}],
  "typography": {
    "headline_font": "string or null",
    "body_font": "string or null",
    "notes": "string"
  },
  "voice_and_tone": "1-2 sentence description",
  "logo_rules": "string (clear space, minimum size, do-not-modify rules, etc.)",
  "imagery_style": "string (photography style, treatment, mood)",
  "dos": ["..."],
  "donts": ["..."]
}

If a field is not specified in the document, use null or an empty list. Do
not invent values."""

        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part.from_bytes(data=pdf_bytes, mime_type=mime),
                extraction_prompt,
            ],
        )

        # Strip code fences if the model added them despite instructions.
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("`").strip()

        brand = json.loads(text)
        # Persist on session state so downstream agents see it.
        tool_context.state["brand"] = brand
        return {"status": "success", "brand": brand}

    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"Brand JSON parse failed: {e}. Raw text: {text[:300]}",
        }
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


# Google Ads Responsive Display asset specs.
# Aspect ratios are what Google requires; we render at these pixel sizes
# (above the minimums, under the 5MB limit).
RDA_ASSETS = [
    {
        "key": "marketing_image_landscape",
        "ratio": "1.91:1",
        "size": (1200, 628),
        "role": "Marketing image — landscape",
        "use_vto": True,        # composite the VTO photo into this asset
    },
]


# Social media asset specs (Instagram + TikTok). All use the VTO photo
# as hero; Nano Banana reframes for each aspect ratio.
SOCIAL_ASSETS = {
    "instagram_feed": {
        "ratio": "4:5",
        "size": (1080, 1350),
        "role": "Instagram feed post (portrait)",
        "platform": "Instagram",
        "placement": "feed",
        "safe_zone": "Center 60% — assume profile chrome bottom-left and "
                     "action icons right edge.",
    },
    "instagram_reels_story": {
        "ratio": "9:16",
        "size": (1080, 1920),
        "role": "Instagram Reels / Story (vertical full-screen)",
        "platform": "Instagram",
        "placement": "reels_story",
        "safe_zone": "Center 1080x1420 — Instagram overlays UI in the top "
                     "~250px and bottom ~250px.",
    },
    "tiktok_vertical": {
        "ratio": "9:16",
        "size": (1080, 1920),
        "role": "TikTok vertical post / cover (full-screen)",
        "platform": "TikTok",
        "placement": "feed",
        "safe_zone": "Center 1080x1420 — TikTok overlays username + "
                     "captions in the bottom ~340px and action rail on "
                     "the right ~150px.",
    },
}


async def generate_ad_creative(
    asset_key: str,
    creative_brief: str,
    tool_context: ToolContext,
    vto_artifact_filename: Optional[str] = None,
) -> dict:
    """Generate one Google Ads Responsive Display asset.

    Routing:
      - If the asset uses the VTO photo (`use_vto=True`): call
        Gemini 2.5 Flash Image (Nano Banana) to *edit/composite* the VTO
        person-with-product photo into the branded ad layout. This preserves
        the actual product/person.
      - Otherwise (logos, branded background-only assets): call Imagen 4
        for a clean text-to-image render from the creative brief.

    The result is saved as an artifact AND written to disk, then resized
    to the exact Google-recommended pixel dimensions.

    Args:
        asset_key: Currently only "marketing_image_landscape" (1.91:1,
            1200x628). Determines aspect ratio and which model to use.
        creative_brief: A focused prompt describing the asset's layout,
            where the product/person sits, where text goes, mood,
            background, colors — all on-brand per the brand guidelines.
        vto_artifact_filename: Artifact name for the VTO output image.
            OPTIONAL — if not provided, falls back to
            state['vto_image_filename'] which find_vto_image populated.
            This is the preferred path: the LLM shouldn't pass this arg
            at all. Ignored for logo assets (use_vto=False).

    Returns:
        dict with keys: status, asset_key, artifact_filename, model_used,
        dimensions, message.
    """
    spec = next((a for a in RDA_ASSETS if a["key"] == asset_key), None)
    if spec is None:
        return {
            "status": "error",
            "message": f"Unknown asset_key '{asset_key}'. Must be one of: "
                       f"{[a['key'] for a in RDA_ASSETS]}",
        }

    target_w, target_h = spec["size"]

    # Belt-and-suspenders: scrub explicit color codes/words out of the
    # creative_brief if it's coming in from an LLM-authored creative
    # plan. The deterministic plan builder already omits these, but
    # the LLM fallback path can include "navy", "#1B2D45", etc., and
    # Nano Banana reads those as instructions to RECOLOR THE OUTFIT.
    # Stripping at the boundary stops that failure mode regardless of
    # where the brief came from.
    if spec.get("use_vto"):
        if creative_brief:
            creative_brief = re.sub(
                r"#[0-9A-Fa-f]{3,8}\b", "neutral tones", creative_brief
            )
            creative_brief = re.sub(
                r"\b(brand\s+colors?|primary\s+colors?|secondary\s+colors?|"
                r"color\s+palette|palette)\b",
                "neutral palette",
                creative_brief, flags=re.IGNORECASE,
            )

    try:
        if spec["use_vto"]:
            # --------------------------------------------------------------
            # Nano Banana path: edit the VTO photo into an ad layout.
            # --------------------------------------------------------------
            # Resolution order: same as generate_social_post — never
            # fail just because brand_parser/find_vto_image was skipped
            # or set the cursor instead of the filename.
            #   1. state["vto_image_filename"]  (set by find_vto_image)
            #   2. The arg the LLM passed (last resort, may be wrong)
            #   3. state["latest_hero_image"]   (set by VTO + edit tools)
            #   4. Newest edit_result_* / vto_result_* artifact
            chosen_vto = (
                tool_context.state.get("vto_image_filename")
                or vto_artifact_filename
                or tool_context.state.get("latest_hero_image")
            )
            if not chosen_vto:
                try:
                    names = await tool_context.list_artifacts()
                except Exception:
                    names = []
                for prefix in ("edit_result_", "vto_result_"):
                    for n in reversed(names or []):
                        if n.startswith(prefix):
                            chosen_vto = n
                            break
                    if chosen_vto:
                        break

            if not chosen_vto:
                return {
                    "status": "error",
                    "message": (
                        "No VTO/hero image found in this session. Run "
                        "virtual_try_on first (upload a person photo + "
                        "a product photo and ask to try it on), then "
                        "ask for the ad campaign again."
                    ),
                }

            # Pin into state for the next tool in the pipeline.
            tool_context.state["vto_image_filename"] = chosen_vto

            vto_part = await tool_context.load_artifact(filename=chosen_vto)
            if vto_part is None or vto_part.inline_data is None:
                return {
                    "status": "error",
                    "message": f"VTO artifact '{chosen_vto}' not found.",
                }

            # Why so many "do not" rules: the creative_brief is written
            # by an LLM and inevitably contains styling words like "navy
            # coat", "warm sunset palette", etc. Nano Banana reads those
            # as instructions and ends up changing the outfit color or
            # texture in the source photo — exactly what we don't want.
            # The structure below front-loads the preservation rules and
            # explicitly demotes the brief to BACKGROUND-only guidance.
            edit_prompt = (
                "You are editing the attached photograph. This is a "
                "compositing/reframing task, NOT a generation task.\n\n"
                "PRIORITY 1 — PRESERVE THE SOURCE EXACTLY:\n"
                "- The person's face, identity, hair, skin tone, body "
                "shape, and pose must remain IDENTICAL to the attached "
                "photo.\n"
                "- The clothing/product the person is wearing must remain "
                "IDENTICAL — same color, same fabric, same cut, same "
                "design, same fit, same accessories. Do NOT recolor, "
                "restyle, replace, or 'improve' the outfit in any way.\n"
                "- If anything in the creative brief below would change "
                "the person or the outfit, IGNORE that part of the "
                "brief. The source photo wins.\n\n"
                f"PRIORITY 2 — REFRAME for a Google Display ad at "
                f"{spec['ratio']} aspect ratio ({target_w}x{target_h} "
                f"pixels):\n"
                "- The person and product occupy roughly 55-70% of the "
                "frame and remain clearly recognisable.\n"
                "- The remaining ~30-45% of the frame is the BACKGROUND "
                "area where styling from the creative brief applies.\n\n"
                "PRIORITY 3 — STYLE THE BACKGROUND ONLY using the brief:\n"
                f"{creative_brief}\n\n"
                "HARD RULES (apply to the whole image):\n"
                "- Background must be visually clean — NO text, NO "
                "logos, NO wordmarks, NO extra people. Google Ads "
                "overlays headline text at serve time.\n"
                "- No copyrighted characters, celebrities, or competitor "
                "branding anywhere.\n\n"
                "OUTPUT: a single professional ad-photography image "
                "that looks like the source photo with the SAME person "
                "wearing the SAME outfit, placed in a new on-brand "
                "background environment."
            )

            response = _client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[
                    genai_types.Part.from_bytes(
                        data=vto_part.inline_data.data,
                        mime_type=vto_part.inline_data.mime_type or "image/png",
                    ),
                    edit_prompt,
                ],
            )

            image_bytes = _extract_image_bytes(response)
            model_used = "gemini-2.5-flash-image"

        else:
            # --------------------------------------------------------------
            # Imagen path: text-to-image for the abstract brand-color mark.
            # --------------------------------------------------------------
            # NOTE on brands: Imagen blocks prompts that look like an
            # attempt to recreate a real company's logo. Our brief therefore
            # describes a *generic on-brand mark* using the color palette
            # only, not the brand's actual wordmark. The user's real logo
            # belongs in Google Ads as a separately uploaded asset.
            imagen_prompt = (
                f"{spec['role']}. {creative_brief}\n\n"
                f"HARD RULES:\n"
                f"- This is an abstract geometric mark only — NO TEXT, NO "
                f"WORDS, NO LETTERS, NO NUMBERS anywhere in the image.\n"
                f"- Do NOT depict any specific brand name, real company "
                f"logo, trademarked symbol, celebrity likeness, or "
                f"copyrighted character.\n"
                f"- Centered composition, generous margin (≥20% on each "
                f"side), clean flat background.\n"
                f"- High-resolution, vector-style, professional."
            )

            # Pick the Imagen native ratio that's closest to the target so
            # Pillow's center-crop wastes the least pixels. 4:1 isn't
            # native; 16:9 is the widest available source.
            ratio_map = {
                "1:1":    "1:1",
                "1.91:1": "16:9",
                "4:1":    "16:9",
                "16:9":   "16:9",
                "9:16":   "9:16",
                "3:4":    "3:4",
                "4:3":    "4:3",
            }
            imagen_ratio = ratio_map.get(spec["ratio"], "1:1")

            try:
                imagen_result = _client.models.generate_images(
                    model="imagen-4.0-generate-001",
                    prompt=imagen_prompt,
                    config=genai_types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio=imagen_ratio,
                    ),
                )
            except Exception as e:
                return {
                    "status": "error",
                    "asset_key": asset_key,
                    "message": f"Imagen call failed: {type(e).__name__}: {e}. "
                               f"Common causes: prompt blocked by safety "
                               f"filter (try removing brand-name references), "
                               f"or the model is not available in this region.",
                }

            # Imagen returns an empty list when ALL candidates were filtered.
            gen_imgs = getattr(imagen_result, "generated_images", None) or []
            if not gen_imgs:
                return {
                    "status": "error",
                    "asset_key": asset_key,
                    "message": "Imagen returned no images (all candidates "
                               "filtered by safety system). Try simplifying "
                               "the brief and removing any brand names, "
                               "trademarks, or product names from the prompt.",
                }

            image_bytes = gen_imgs[0].image.image_bytes
            if not image_bytes:
                return {
                    "status": "error",
                    "asset_key": asset_key,
                    "message": "Imagen returned an empty image buffer.",
                }
            model_used = "imagen-4.0-generate-001"

        # ------------------------------------------------------------------
        # Sanity-check the bytes before handing them to Pillow.
        # ------------------------------------------------------------------
        if not image_bytes or len(image_bytes) < 100:
            return {
                "status": "error",
                "asset_key": asset_key,
                "message": f"Generated image is empty or too small "
                           f"({len(image_bytes) if image_bytes else 0} bytes). "
                           f"The model may have refused the request.",
            }

        # ------------------------------------------------------------------
        # Resize to exact Google-recommended dimensions (Pillow).
        # ------------------------------------------------------------------
        final_bytes = _fit_to_size(image_bytes, target_w, target_h)

        # Save as artifact so it renders inline in ADK Web.
        out_name = f"ad_{spec['key']}_{uuid.uuid4().hex[:8]}.png"
        await tool_context.save_artifact(
            filename=out_name,
            artifact=genai_types.Part.from_bytes(
                data=final_bytes, mime_type="image/png"
            ),
        )

        # Also drop to local disk for convenience.
        (_OUTPUT_DIR / out_name).write_bytes(final_bytes)

        # Filename-only markdown. Returning a base64 data URI here would
        # push every image into the next turn's LLM context as part of
        # the function_response history and quickly exceed the 1M-token
        # input limit (one 1080x1920 PNG ≈ 250K-700K tokens base64-encoded).
        ad_markdown = f"![{asset_key} ({target_w}x{target_h})]({out_name})"

        return {
            "status": "success",
            "asset_key": asset_key,
            "artifact_filename": out_name,
            "image_markdown": ad_markdown,
            "model_used": model_used,
            "dimensions": f"{target_w}x{target_h}",
            "ratio": spec["ratio"],
            "file_size_bytes": len(final_bytes),
        }

    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


async def generate_social_post(
    platform_key: str,
    visual_brief: str,
    caption: str,
    hashtags: list[str],
    tool_context: ToolContext,
) -> dict:
    """Generate one social media post (image + caption) for the VTO look.

    Uses the same VTO photo (from state['vto_image_filename']) as the hero
    and reframes it for the platform's required aspect ratio with Nano
    Banana. The caption and hashtags are persisted alongside the image
    in a structured artifact so the user gets a complete post they can
    paste into Instagram or TikTok.

    Args:
        platform_key: One of "instagram_feed" (4:5, 1080x1350),
            "instagram_reels_story" (9:16, 1080x1920), or
            "tiktok_vertical" (9:16, 1080x1920).
        visual_brief: How to recompose the VTO photo for THIS platform.
            Should describe framing, mood, background, and what content
            to keep inside the safe zone.
        caption: The post copy. Platform-appropriate length and tone
            (Instagram tolerates longer; TikTok prefers a punchy hook).
        hashtags: List of hashtag strings (without the leading #). The
            tool will format them. 5-10 hashtags is the recommended range
            for both platforms in 2026.

    Returns:
        dict with keys: status, platform_key, image_artifact_filename,
        caption, hashtags, post_text_markdown, dimensions, message.
    """
    spec = SOCIAL_ASSETS.get(platform_key)
    if spec is None:
        return {
            "status": "error",
            "message": f"Unknown platform_key '{platform_key}'. Must be one "
                       f"of: {list(SOCIAL_ASSETS.keys())}",
        }

    target_w, target_h = spec["size"]

    # Same color-scrub as generate_ad_creative: strip hex codes and
    # palette-mention phrases from the visual_brief so Nano Banana
    # doesn't recolor the outfit when "build social posts" runs.
    if visual_brief:
        visual_brief = re.sub(
            r"#[0-9A-Fa-f]{3,8}\b", "neutral tones", visual_brief
        )
        visual_brief = re.sub(
            r"\b(brand\s+colors?|primary\s+colors?|secondary\s+colors?|"
            r"color\s+palette|palette)\b",
            "neutral palette",
            visual_brief, flags=re.IGNORECASE,
        )

    try:
        # Resolution order (mirror find_vto_image's logic so this tool
        # never fails just because brand_parser was skipped or set the
        # cursor instead of the filename):
        #   1. state["vto_image_filename"]  — set by find_vto_image.
        #   2. state["latest_hero_image"]   — set by virtual_try_on /
        #                                     edit_image at the moment
        #                                     they produce an image.
        #   3. Newest edit_result_* artifact in this session.
        #   4. Newest vto_result_* artifact in this session.
        chosen_vto = (
            tool_context.state.get("vto_image_filename")
            or tool_context.state.get("latest_hero_image")
        )
        if not chosen_vto:
            try:
                names = await tool_context.list_artifacts()
            except Exception:
                names = []
            for prefix in ("edit_result_", "vto_result_"):
                for n in reversed(names or []):
                    if n.startswith(prefix):
                        chosen_vto = n
                        break
                if chosen_vto:
                    break

        if not chosen_vto:
            return {
                "status": "error",
                "message": (
                    "No VTO/hero image found in this session. Run "
                    "virtual_try_on first (upload a person photo + a "
                    "product photo and ask to try it on), then ask for "
                    "social posts again."
                ),
            }

        # Pin it back into state so subsequent calls in the same
        # session pick it up via the fast path.
        tool_context.state["vto_image_filename"] = chosen_vto

        vto_part = await tool_context.load_artifact(filename=chosen_vto)
        if vto_part is None or vto_part.inline_data is None:
            return {
                "status": "error",
                "message": f"VTO artifact '{chosen_vto}' not found.",
            }

        # Same prompt structure as the ad tool: front-load preservation
        # rules and demote the visual_brief to background-only guidance.
        # Without this, the LLM-written visual_brief contains outfit
        # styling words that Nano Banana applies to the actual outfit,
        # so e.g. asked for an Instagram post the model would recolor
        # the clothes to "match the curated magazine palette". The
        # source photo must win.
        edit_prompt = (
            "You are editing the attached photograph. This is a "
            "compositing/reframing task, NOT a generation task.\n\n"
            "PRIORITY 1 — PRESERVE THE SOURCE EXACTLY:\n"
            "- The person's face, identity, hair, skin tone, body "
            "shape, and pose must remain IDENTICAL to the attached "
            "photo.\n"
            "- The clothing/product the person is wearing must remain "
            "IDENTICAL — same color, same fabric, same cut, same "
            "design, same fit, same accessories. Do NOT recolor, "
            "restyle, replace, or 'improve' the outfit.\n"
            "- If anything in the visual brief below would change "
            "the person or the outfit, IGNORE that part. The source "
            "photo wins.\n\n"
            f"PRIORITY 2 — REFRAME for {spec['platform']} "
            f"{spec['placement']} at {spec['ratio']} aspect ratio "
            f"({target_w}x{target_h} pixels):\n"
            f"- Keep the hero subject inside the platform safe zone: "
            f"{spec['safe_zone']}\n"
            "- The hero subject is the person + outfit from the "
            "attached photo. The rest of the frame is the BACKGROUND "
            "area where the visual brief applies.\n\n"
            "PRIORITY 3 — STYLE THE BACKGROUND ONLY using the brief:\n"
            f"{visual_brief}\n\n"
            "HARD RULES (apply to the whole image):\n"
            "- Do NOT render any text, logos, captions, watermarks, "
            "or wordmarks. Captions are added by the social platform UI.\n"
            f"- Make the result feel like a native {spec['platform']} "
            "post — premium but organic, not stiff or ad-like.\n\n"
            f"OUTPUT: a single high-quality {spec['platform']} post "
            f"image — the same person wearing the same outfit, placed "
            f"in a new on-brand environment."
        )

        response = _client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                genai_types.Part.from_bytes(
                    data=vto_part.inline_data.data,
                    mime_type=vto_part.inline_data.mime_type or "image/png",
                ),
                edit_prompt,
            ],
        )

        image_bytes = _extract_image_bytes(response)
        if not image_bytes or len(image_bytes) < 100:
            return {
                "status": "error",
                "platform_key": platform_key,
                "message": "Generated image is empty or too small. The "
                           "model may have refused the request.",
            }

        final_bytes = _fit_to_size(image_bytes, target_w, target_h)

        # Save the image as an artifact. We do NOT save the caption as a
        # separate text artifact — that tempts the agent's reply to
        # reference the .txt filename ("Caption: text/plain") instead of
        # showing the actual copy. The caption belongs in chat, in the
        # post_text_markdown below.
        suffix = uuid.uuid4().hex[:8]
        img_name = f"social_{platform_key}_{suffix}.png"
        await tool_context.save_artifact(
            filename=img_name,
            artifact=genai_types.Part.from_bytes(
                data=final_bytes, mime_type="image/png"
            ),
        )
        (_OUTPUT_DIR / img_name).write_bytes(final_bytes)

        hashtag_str = " ".join(f"#{h.lstrip('#')}" for h in (hashtags or []))
        full_caption = f"{caption}\n\n{hashtag_str}".strip()
        # Also drop a .txt next to the image on local disk so the user
        # can grab it without parsing chat output. NOT saved as an
        # artifact (see comment above).
        txt_name = f"social_{platform_key}_{suffix}.txt"
        (_OUTPUT_DIR / txt_name).write_text(full_caption)

        # The post_text_markdown contains the caption verbatim inline.
        # The agent must paste this block as-is in its final reply so
        # the user sees real copy, not a filename reference.
        post_markdown = (
            f"**{spec['platform']} — {spec['placement']}** "
            f"({target_w}x{target_h})\n\n"
            f"![{spec['platform']} post]({img_name})\n\n"
            f"**Caption:**\n\n{caption}\n\n"
            f"**Hashtags:** {hashtag_str}"
        )

        return {
            "status": "success",
            "platform_key": platform_key,
            "image_artifact_filename": img_name,
            "caption": caption,
            "hashtags": hashtag_str,
            "post_text_markdown": post_markdown,
            "dimensions": f"{target_w}x{target_h}",
            "ratio": spec["ratio"],
            "platform": spec["platform"],
            "placement": spec["placement"],
            "file_size_bytes": len(final_bytes),
        }

    except Exception as e:
        return {"status": "error",
                "platform_key": platform_key,
                "message": f"{type(e).__name__}: {e}"}


async def find_brand_pdf(tool_context: ToolContext) -> dict:
    """Find the brand guidelines PDF for this session.

    Resolution order:
      1. state['brand_pdf_filename'] if already set — usually by the
         _prefetch_brand_pdf_from_gcs callback that ran before this tool.
      2. Newest PDF artifact in the session (a user attachment via
         SaveFilesAsArtifactsPlugin will appear here).

    Returns:
        dict with keys: status, pdf_artifact_filename, source
        ("cursor" | "scan"), all_pdfs (when scanning), message.
    """
    # 1. State cursor.
    cursor = tool_context.state.get("brand_pdf_filename")
    if cursor:
        try:
            part = await tool_context.load_artifact(filename=cursor)
        except Exception:
            part = None
        if part is not None:
            return {"status": "success",
                    "pdf_artifact_filename": cursor,
                    "source": "cursor"}
        # Cursor stale; fall through to scan.

    # 2. Scan session artifacts.
    try:
        names = await tool_context.list_artifacts()
    except Exception as e:
        return {"status": "error",
                "message": f"Could not list artifacts: {e}"}

    pdfs = []
    for n in names or []:
        try:
            part = await tool_context.load_artifact(filename=n)
        except Exception:
            continue
        mime = (part and part.inline_data and part.inline_data.mime_type) or ""
        if mime == "application/pdf" or n.lower().endswith(".pdf"):
            pdfs.append(n)

    if not pdfs:
        gcs_err = tool_context.state.get("_brand_pdf_gcs_error")
        if BRAND_PDF_BUCKET and gcs_err:
            return {"status": "error",
                    "message": f"BRAND_PDF_BUCKET is set to "
                               f"'{BRAND_PDF_BUCKET}' but loading from GCS "
                               f"failed: {gcs_err}. Check the bucket name, "
                               f"ensure the runtime service account has "
                               f"roles/storage.objectViewer, or attach a "
                               f"PDF in chat instead."}
        if BRAND_PDF_BUCKET:
            return {"status": "error",
                    "message": f"BRAND_PDF_BUCKET is set to "
                               f"'{BRAND_PDF_BUCKET}' but no .pdf files "
                               f"were found in "
                               f"gs://{BRAND_PDF_BUCKET}/"
                               f"{BRAND_PDF_PREFIX or ''}. Upload a brand "
                               f"guidelines PDF to that bucket, or attach "
                               f"one in chat."}
        return {"status": "error",
                "message": "BRAND_PDF_BUCKET environment variable is not "
                           "set, and no PDF was attached in chat. Either "
                           "attach the brand guidelines PDF in this "
                           "message, or redeploy the agent with "
                           "--brand-pdf-bucket=YOUR_BUCKET so it can "
                           "auto-pull from GCS."}

    chosen = pdfs[-1]  # most recently saved
    tool_context.state["brand_pdf_filename"] = chosen
    return {"status": "success",
            "pdf_artifact_filename": chosen,
            "source": "scan",
            "all_pdfs": pdfs}


# ---------------------------------------------------------------------------
# Brand PDF source: Google Cloud Storage
# ---------------------------------------------------------------------------
# Customers can drop brand guidelines into a GCS bucket instead of
# attaching the PDF to the chat. Set BRAND_PDF_BUCKET to the bucket name
# (without the `gs://` prefix) to enable the GCS-first flow.
#
#   BRAND_PDF_BUCKET=my-brand-pdfs
#   BRAND_PDF_PREFIX=brands/            (optional subdirectory inside the bucket)
#
# The bucket can be public (anyone with the URL can read) or readable by
# the agent's service account. For Agent Engine deployments, ensure the
# engine's runtime service account has `roles/storage.objectViewer` on
# the bucket.
BRAND_PDF_BUCKET = os.getenv("BRAND_PDF_BUCKET", "").strip()
BRAND_PDF_PREFIX = os.getenv("BRAND_PDF_PREFIX", "").strip()

# Log on import so the deployed-container startup logs make it obvious
# whether the GCS path is active. Without this, a missing env var
# manifests only as "couldn't find brand guidelines" downstream and
# is hard to diagnose without code inspection.
if BRAND_PDF_BUCKET:
    print(f"[marketing_campaign_agent] BRAND_PDF_BUCKET=gs://"
          f"{BRAND_PDF_BUCKET}/{BRAND_PDF_PREFIX or ''} — GCS auto-pull "
          f"enabled.")
else:
    print("[marketing_campaign_agent] BRAND_PDF_BUCKET not set — "
          "agent will only use chat-attached PDFs. To enable GCS "
          "auto-pull, redeploy with --brand-pdf-bucket=YOUR_BUCKET.")


def _gcs_client():
    """Lazy-import google-cloud-storage so the dependency isn't required
    for users who only attach PDFs in chat."""
    from google.cloud import storage  # noqa: WPS433  (lazy import is intentional)
    return storage.Client()


async def list_brand_pdfs_in_gcs(tool_context: ToolContext) -> dict:
    """List the brand PDFs available in the configured GCS bucket.

    Reads the bucket name from env var BRAND_PDF_BUCKET (and optional
    BRAND_PDF_PREFIX for a sub-folder). Returns the .pdf object names so
    the agent or downstream tool can pick one.

    Returns:
        dict with keys:
          status: "success" | "error"
          bucket, prefix (when success)
          pdfs: list of {name, updated, size_bytes} (newest first)
          message: str (when error)
    """
    if not BRAND_PDF_BUCKET:
        return {"status": "error",
                "message": "BRAND_PDF_BUCKET env var is not set. Fall "
                           "back to find_brand_pdf for attached PDFs."}

    try:
        client = _gcs_client()
        bucket = client.bucket(BRAND_PDF_BUCKET)
        # list_blobs is sync; that's fine inside a tool — ADK invokes
        # tools on a worker thread.
        blobs = list(bucket.list_blobs(
            prefix=BRAND_PDF_PREFIX or None,
            # max_results keeps the listing snappy for big buckets; if
            # the customer has > 200 brand PDFs they can override.
            max_results=200,
        ))
    except Exception as e:
        return {"status": "error",
                "message": f"GCS list failed for bucket "
                           f"'{BRAND_PDF_BUCKET}': {type(e).__name__}: {e}"}

    pdfs = []
    for b in blobs:
        if not b.name.lower().endswith(".pdf"):
            continue
        pdfs.append({
            "name": b.name,
            "updated": b.updated.isoformat() if b.updated else None,
            "size_bytes": b.size or 0,
        })

    if not pdfs:
        return {"status": "error",
                "message": f"No .pdf objects found in "
                           f"gs://{BRAND_PDF_BUCKET}/{BRAND_PDF_PREFIX}."}

    # Newest first — most customers want the latest brand book.
    pdfs.sort(key=lambda p: p["updated"] or "", reverse=True)
    return {"status": "success",
            "bucket": BRAND_PDF_BUCKET,
            "prefix": BRAND_PDF_PREFIX,
            "pdfs": pdfs}


async def fetch_brand_pdf_from_gcs(
    object_name: str,
    tool_context: ToolContext,
) -> dict:
    """Download a brand PDF from the configured GCS bucket and save it
    as a session artifact.

    The downloaded PDF gets the same lifecycle as a user-uploaded one:
    saved via `tool_context.save_artifact`, then read by the existing
    `extract_brand_guidelines` tool. No other code path needs to know
    whether the PDF came from the user or from GCS.

    Args:
        object_name: The full GCS object name (e.g. "brands/abercrombie.pdf").
            Use list_brand_pdfs_in_gcs first to discover the names.

    Returns:
        dict with keys: status, pdf_artifact_filename (the local artifact
        name; same field name as find_brand_pdf for compatibility), message.
    """
    if not BRAND_PDF_BUCKET:
        return {"status": "error",
                "message": "BRAND_PDF_BUCKET env var is not set."}
    if not object_name or not object_name.lower().endswith(".pdf"):
        return {"status": "error",
                "message": f"object_name '{object_name}' is not a .pdf"}

    try:
        client = _gcs_client()
        bucket = client.bucket(BRAND_PDF_BUCKET)
        blob = bucket.blob(object_name)
        if not blob.exists():
            return {"status": "error",
                    "message": f"gs://{BRAND_PDF_BUCKET}/{object_name} "
                               f"does not exist or is not readable."}
        pdf_bytes = blob.download_as_bytes()
    except Exception as e:
        return {"status": "error",
                "message": f"GCS download failed for "
                           f"gs://{BRAND_PDF_BUCKET}/{object_name}: "
                           f"{type(e).__name__}: {e}"}

    if not pdf_bytes or len(pdf_bytes) < 100:
        return {"status": "error",
                "message": f"Downloaded PDF is empty or too small "
                           f"({len(pdf_bytes)} bytes)."}

    # Use just the basename for the artifact filename so chat references
    # are concise. e.g. "brands/abercrombie.pdf" → "abercrombie.pdf".
    artifact_name = Path(object_name).name
    await tool_context.save_artifact(
        filename=artifact_name,
        artifact=genai_types.Part.from_bytes(
            data=pdf_bytes, mime_type="application/pdf"
        ),
    )

    tool_context.state["brand_pdf_filename"] = artifact_name
    tool_context.state["brand_pdf_source"] = (
        f"gs://{BRAND_PDF_BUCKET}/{object_name}"
    )
    return {"status": "success",
            "pdf_artifact_filename": artifact_name,
            "gcs_source": f"gs://{BRAND_PDF_BUCKET}/{object_name}",
            "size_bytes": len(pdf_bytes)}


async def find_vto_image(tool_context: ToolContext) -> dict:
    """Find the most recent VTO / edited image to use as the campaign hero.

    Resolution order (most reliable first):

      1. **state['latest_hero_image']** — set by `virtual_try_on` and
         `edit_image` each time they succeed. This is the authoritative
         cursor: if the user did a try-on, then an edit, this points at
         the edit's output. Used by both the marketing campaign agent
         and the social media agent so they always operate on the most
         recent image, not whatever the user uploaded first.

      2. Newest `edit_result_*` artifact (edited > raw VTO when both
         exist, because an edit is always downstream of a VTO).

      3. Newest `vto_result_*` artifact.

      4. Newest non-PDF image artifact (a directly uploaded photo).

    The chosen filename is also written to state['vto_image_filename']
    for backward compatibility with generate_ad_creative and
    generate_social_post.

    Returns:
        dict with keys: status, vto_image_filename, source
        ("cursor" | "edit" | "vto" | "fallback"), all_candidates,
        message.
    """
    # Pull the current artifact list once; we'll use it for cursor
    # invalidation AND fallback scanning.
    try:
        names = await tool_context.list_artifacts()
    except Exception as e:
        return {"status": "error",
                "message": f"Could not list artifacts: {e}"}
    names = names or []

    # 1. Trust the explicit cursor IF (a) it still loads as an image
    # AND (b) no newer non-result image upload has appeared since.
    # Condition (b) catches the case where the user reused a session,
    # uploaded a fresh photo, then asked for a new campaign — without
    # it, the agent would silently pick up the previous session's VTO
    # from the cursor and ignore the fresh upload.
    cursor = tool_context.state.get("latest_hero_image")
    if cursor and cursor in names:
        # Find the cursor's position in the save-order list. Anything
        # after it that's a fresh upload (not a tool output) indicates
        # the user wants a new run.
        try:
            cursor_idx = names.index(cursor)
        except ValueError:
            cursor_idx = -1

        newer_upload = None
        for n in names[cursor_idx + 1:]:
            if (n.startswith("vto_result_") or n.startswith("edit_result_")
                    or n.startswith("ad_") or n.startswith("social_")
                    or n.lower().endswith(".pdf")
                    or n.lower().endswith(".txt")):
                continue
            # Confirm it's actually an image artifact.
            try:
                part = await tool_context.load_artifact(filename=n)
            except Exception:
                continue
            mime = (part and part.inline_data
                    and part.inline_data.mime_type) or ""
            if mime.startswith("image/"):
                newer_upload = n
                break

        if newer_upload is None:
            # Cursor is still authoritative.
            try:
                part = await tool_context.load_artifact(filename=cursor)
            except Exception:
                part = None
            if part is not None:
                tool_context.state["vto_image_filename"] = cursor
                return {
                    "status": "success",
                    "vto_image_filename": cursor,
                    "source": "cursor",
                    "all_candidates": [cursor],
                }
        # Else: a newer upload exists. Clear the stale cursor and
        # restart the campaign fresh from this upload. The user will
        # need to run virtual_try_on on the new upload — emit an error
        # explaining that rather than silently using the old VTO.
        #
        # NOTE: ADK's State object has no .pop() and no __delitem__.
        # The only mutation primitives are __setitem__, update, and
        # setdefault. So we "clear" by setting to None — downstream
        # `if cursor:` checks treat None and missing identically.
        tool_context.state["latest_hero_image"] = None
        tool_context.state["vto_image_filename"] = None
        return {
            "status": "error",
            "message": (
                f"A new image '{newer_upload}' has been uploaded since "
                f"the last virtual try-on. Run virtual_try_on again "
                f"with the new image so the campaign uses the fresh "
                f"hero photo. (Previous hero '{cursor}' was discarded "
                f"to avoid using stale data from an earlier session.)"
            ),
        }

    def _newest(prefix: str) -> Optional[str]:
        for n in reversed(names):
            if n.startswith(prefix):
                return n
        return None

    # 2 & 3. Edited image takes precedence over raw VTO output because
    # an edit is downstream of a try-on. If the user did edit-after-VTO
    # but the state cursor was somehow lost, we still want the edit.
    chosen = _newest("edit_result_")
    source = "edit"
    if not chosen:
        chosen = _newest("vto_result_")
        source = "vto"
    if not chosen:
        # 4. Last resort: newest non-PDF, non-ad image. Likely a raw upload.
        for n in reversed(names):
            if (n.startswith("ad_") or n.startswith("social_")
                    or n.lower().endswith(".pdf") or n.lower().endswith(".txt")):
                continue
            try:
                part = await tool_context.load_artifact(filename=n)
            except Exception:
                continue
            mime = (part and part.inline_data
                    and part.inline_data.mime_type) or ""
            if mime.startswith("image/"):
                chosen = n
                source = "fallback"
                break

    if not chosen:
        return {"status": "error",
                "message": "No VTO image found in the session. Run the "
                           "virtual_try_on tool first (upload a person "
                           "photo + product photo) before asking for a "
                           "campaign."}

    tool_context.state["vto_image_filename"] = chosen
    # Update the cursor so subsequent calls in this session are fast.
    tool_context.state["latest_hero_image"] = chosen
    return {"status": "success",
            "vto_image_filename": chosen,
            "source": source,
            "all_candidates": names}


async def save_creative_plan(
    campaign_concept: str,
    marketing_image_landscape: str,
    tool_context: ToolContext,
) -> dict:
    """Persist the creative brief to session state.

    The creative_director calls this instead of trying to emit a JSON
    object as its text reply (Gemini frequently truncates or returns
    empty content when asked to do that).

    The VTO image filename is NOT an argument here. It's read from
    state['vto_image_filename'] (set by find_vto_image earlier in the
    pipeline) so the LLM cannot accidentally pass a wrong filename and
    cause the image model to ignore the hero photo.

    Args:
        campaign_concept: One sentence describing the overall idea.
        marketing_image_landscape: Brief for the 1.91:1 (1200x628)
            marketing image. Must use the VTO photo as hero.

    Returns:
        dict with keys: status, message.
    """
    vto_filename = tool_context.state.get("vto_image_filename")
    if not vto_filename:
        return {"status": "error",
                "message": "No vto_image_filename in state. The "
                           "find_vto_image step must run first."}

    plan = {
        "vto_artifact_filename": vto_filename,
        "campaign_concept": campaign_concept,
        "briefs": {
            "marketing_image_landscape": marketing_image_landscape,
        },
    }
    tool_context.state["creative_plan"] = plan
    return {"status": "success",
            "message": f"Plan saved. Hero image: {vto_filename}. "
                       f"Concept: {campaign_concept}"}


async def get_creative_plan(tool_context: ToolContext) -> dict:
    """Return the creative plan written by the creative_director.

    Reads `state['creative_plan']`. The plan is stored as a dict by
    save_creative_plan, but for robustness this also handles a JSON
    string (in case the planner agent wrote one directly).

    Returns:
        dict with keys:
          status: "success" | "error"
          vto_artifact_filename, campaign_concept, briefs  (when success)
          message: str  (when error)
    """
    raw = tool_context.state.get("creative_plan")
    if raw is None:
        return {"status": "error",
                "message": "No creative_plan in session state. The "
                           "creative_director step did not run or failed."}

    if isinstance(raw, dict):
        plan = raw
    else:
        text = str(raw).strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("`").strip()
        try:
            plan = json.loads(text)
        except json.JSONDecodeError as e:
            return {"status": "error",
                    "message": f"Could not parse creative_plan JSON: {e}"}

    required = {"vto_artifact_filename", "campaign_concept", "briefs"}
    missing = required - set(plan)
    if missing:
        return {"status": "error",
                "message": f"creative_plan missing keys: {sorted(missing)}"}

    return {"status": "success", **plan}


def _extract_image_bytes(response) -> bytes:
    """Pull the first inline image part out of a Gemini multimodal response.

    Gemini Nano Banana can return a response with only a text part
    (refusal / explanation) instead of an image when the prompt trips
    a safety filter. In that case we raise a descriptive error instead
    of returning a misleading "no image" exception.
    """
    text_parts = []
    for cand in response.candidates or []:
        for part in cand.content.parts or []:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                return part.inline_data.data
            if getattr(part, "text", None):
                text_parts.append(part.text)
    note = (" Model said: " + " ".join(text_parts)[:200]) if text_parts else ""
    raise RuntimeError(
        "Model returned no image data — likely blocked by safety filter "
        "or content policy." + note
    )


def _fit_to_size(image_bytes: bytes, target_w: int, target_h: int) -> bytes:
    """Resize/center-crop an image to exactly (target_w, target_h) PNG.

    Google's Responsive Display landscape asset is 1200x628 (1.91:1), square
    is 1200x1200, etc. The image models won't hit these exact pixels, so we
    do a center-crop to the target aspect ratio, then resize to exact dims.
    Raises a descriptive ValueError if the input bytes aren't decodable.
    """
    from PIL import Image, UnidentifiedImageError  # local: keep cold-start cheap

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        # Pillow couldn't make sense of the buffer. Most common cause is
        # an empty / truncated response from the image model.
        preview = image_bytes[:40] if image_bytes else b""
        raise ValueError(
            f"Could not decode generated image "
            f"({len(image_bytes) if image_bytes else 0} bytes, "
            f"starts with: {preview!r}). The image model likely returned "
            f"a non-image response (refusal, filter, or error)."
        ) from e
    target_ratio = target_w / target_h
    src_ratio = img.width / img.height

    if src_ratio > target_ratio:
        # Source is wider than target — crop sides.
        new_w = int(img.height * target_ratio)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        # Source is taller — crop top/bottom.
        new_h = int(img.width / target_ratio)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    out = buf.getvalue()

    # Google caps Responsive Display assets at 5MB. PNG should be well under.
    if len(out) > 5 * 1024 * 1024:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        out = buf.getvalue()
    return out


# ===========================================================================
# Sub-agents (the pipeline)
# ===========================================================================

_BRAND_PARSER_INSTRUCTION = """\
ABSOLUTE RULES:
- Use FUNCTION CALLS only. Never write code. Never emit a code block.
  Never use Python syntax like print(), default_api, triple quotes, or
  raw string prefixes.
- Pass each argument as a plain string.
- Do all the steps in this single turn.

STEP 1 — Locate the VTO hero image.
  Call: find_vto_image()
  If the result's status is "error", reply with the message and STOP —
  there's nothing to build a campaign around without a hero image.

STEP 2 — Get the brand PDF (OPTIONAL — the pipeline can run without it).

  The framework may have already pre-loaded a PDF from GCS into state
  under the key 'brand_pdf_filename'. The find_brand_pdf tool will
  return that filename whether the PDF came from a GCS prefetch or a
  user attachment.

  Call: find_brand_pdf()

  If find_brand_pdf returns status="success":
    Call: extract_brand_guidelines(
        pdf_artifact_filename=<the pdf_artifact_filename from step 2>)
    If extraction succeeds, send a SHORT reply (2-3 sentences) summarizing
    the brand: name, primary colors as hex, voice. If the PDF came from
    GCS (state has 'brand_pdf_source' starting with 'gs://'), mention
    the source briefly. STOP.

  If find_brand_pdf returns status="error":
    DO NOT stop. The campaign can still run with neutral defaults.
    Send a SHORT reply: "No brand guidelines found — proceeding with
    neutral defaults. (To get on-brand styling, attach a brand PDF or
    set BRAND_PDF_BUCKET.)"  Then STOP. The next agent will use
    sensible fallback brand values.

  Do not call list_brand_pdfs_in_gcs or fetch_brand_pdf_from_gcs
  unless the user has explicitly asked to pick a specific brand by
  name. The framework's GCS prefetch callback already handled
  auto-pull before this step ran.
"""


async def _prefetch_brand_pdf_from_gcs(
    callback_context: CallbackContext,
    llm_request,  # noqa: ARG001 — typed positionally, may be unused
) -> Optional[LlmResponse]:
    """Auto-load the brand PDF from GCS before the brand_parser LLM runs.

    Why a callback: the LLM has been unreliable about choosing the right
    PDF when multiple exist. With this callback, if BRAND_PDF_BUCKET is
    set and the session doesn't already have a brand_pdf_filename, we
    pick the newest PDF in the bucket and download it deterministically.
    The brand_parser LLM then only has to call extract_brand_guidelines.

    Returns None always (we never short-circuit the LLM) — the goal is
    only to populate state['brand_pdf_filename'] before the LLM picks
    its tool calls.
    """
    state = callback_context.state
    if state.get("brand_pdf_filename"):
        return None  # already chosen, nothing to do
    if not BRAND_PDF_BUCKET:
        return None  # GCS not configured; let find_brand_pdf path run

    # PREFERENCE: chat-attached PDFs WIN over GCS auto-pull.
    #
    # If the user uploaded a PDF in chat ("create posts based on the
    # guidelines I attached here"), the SaveFilesAsArtifactsPlugin has
    # already saved it as a session artifact. Don't override that with
    # a GCS pull — the user explicitly chose a specific PDF.
    try:
        existing = await callback_context.list_artifacts()
    except Exception:
        existing = []
    for n in existing or []:
        if n.lower().endswith(".pdf"):
            # Found a user-attached PDF. Skip GCS; find_brand_pdf will
            # discover this same artifact and use it.
            state["brand_pdf_source"] = "attachment"
            return None

    # Lazy GCS fetch. We can't await tools through callback_context
    # directly, but the storage client itself works fine here.
    try:
        client = _gcs_client()
        bucket = client.bucket(BRAND_PDF_BUCKET)
        blobs = [
            b for b in bucket.list_blobs(
                prefix=BRAND_PDF_PREFIX or None, max_results=200,
            )
            if b.name.lower().endswith(".pdf")
        ]
    except Exception as e:
        # Silent failure: fall through to user-attached PDF path.
        state["_brand_pdf_gcs_error"] = (
            f"{type(e).__name__}: {e}"
        )
        return None

    if not blobs:
        return None

    # Pick the newest blob by updated timestamp.
    blobs.sort(key=lambda b: b.updated or "", reverse=True)
    chosen = blobs[0]
    object_name = chosen.name

    try:
        pdf_bytes = chosen.download_as_bytes()
    except Exception as e:
        state["_brand_pdf_gcs_error"] = (
            f"download failed: {type(e).__name__}: {e}"
        )
        return None

    if not pdf_bytes or len(pdf_bytes) < 100:
        return None

    artifact_name = Path(object_name).name
    try:
        await callback_context.save_artifact(
            filename=artifact_name,
            artifact=genai_types.Part.from_bytes(
                data=pdf_bytes, mime_type="application/pdf"
            ),
        )
    except Exception as e:
        state["_brand_pdf_gcs_error"] = (
            f"save_artifact failed: {type(e).__name__}: {e}"
        )
        return None

    state["brand_pdf_filename"] = artifact_name
    state["brand_pdf_source"] = f"gs://{BRAND_PDF_BUCKET}/{object_name}"
    return None


def _make_brand_parser(name: str = "brand_parser") -> Agent:
    """Create a fresh brand_parser Agent instance.

    Each SequentialAgent that needs brand context gets its own instance —
    ADK agents have a single parent, so the same instance can't appear
    in two sub_agents lists. Each call to this factory returns a new
    Agent object so each pipeline gets a distinct parent-child link.
    """
    return Agent(
        name=name,
        model="gemini-2.5-flash",
        description=(
            "Locates the VTO hero image and reads the customer's brand "
            "guidelines PDF (from GCS if BRAND_PDF_BUCKET is set, "
            "otherwise from chat attachments), then extracts a "
            "structured brand profile."
        ),
        instruction=_BRAND_PARSER_INSTRUCTION,
        tools=[
            find_vto_image,
            list_brand_pdfs_in_gcs,
            fetch_brand_pdf_from_gcs,
            find_brand_pdf,
            extract_brand_guidelines,
        ],
        # Auto-pulls the newest PDF from GCS before the LLM runs, so the
        # LLM's job in the common case is just to call
        # extract_brand_guidelines. Falls through (returns None) when
        # GCS is not configured or no PDFs are present.
        before_model_callback=_prefetch_brand_pdf_from_gcs,
        output_key="brand_summary",
    )


# One instance per pipeline.
brand_parser_agent = _make_brand_parser("brand_parser")
brand_parser_agent_for_social = _make_brand_parser("brand_parser_for_social")


def _hex_list(colors) -> list[str]:
    """Normalize a primary/secondary_colors entry from brand JSON.

    Brand profiles may store colors as [{"name":..,"hex":..}], or as
    bare ["#RRGGBB", ...], or be missing. Return a flat list of hex
    codes; empty if nothing usable.
    """
    out: list[str] = []
    for item in colors or []:
        if isinstance(item, dict):
            h = item.get("hex") or item.get("value")
            if h:
                out.append(str(h))
        elif isinstance(item, str):
            out.append(item)
    return out


async def _resolve_hero_image(callback_context) -> Optional[str]:
    """Resolve the hero image filename using the same chain as the
    image-producing tools, so the plan builder doesn't fail just
    because brand_parser's LLM skipped find_vto_image.

    Order:
      1. state["vto_image_filename"]
      2. state["latest_hero_image"]
      3. Newest edit_result_* in session artifacts
      4. Newest vto_result_* in session artifacts

    Returns the filename or None if nothing is in the session.
    """
    state = callback_context.state
    chosen = (
        state.get("vto_image_filename")
        or state.get("latest_hero_image")
    )
    if chosen:
        return chosen

    try:
        names = await callback_context.list_artifacts()
    except Exception:
        return None
    for prefix in ("edit_result_", "vto_result_"):
        for n in reversed(names or []):
            if n.startswith(prefix):
                return n
    return None


async def _build_creative_plan_from_state(callback_context) -> Optional[dict]:
    """Construct the creative plan deterministically from session state.

    Returns the plan dict (also written into state['creative_plan']
    and state['vto_image_filename']) or None if no hero image can be
    found in the session at all.
    """
    state = callback_context.state
    vto = await _resolve_hero_image(callback_context)
    if not vto:
        return None
    # Pin the resolved filename back to state so downstream tools
    # take the fast path.
    state["vto_image_filename"] = vto

    brand = state.get("brand") or {}
    if not brand:
        # Generic plan when no brand profile was parsed — better than
        # failing the whole pipeline. Captions will use neutral tone.
        brand = {}

    brand_name = brand.get("brand_name") or "the brand"
    voice = brand.get("voice_and_tone") or "modern and aspirational"
    imagery = brand.get("imagery_style") or "clean, premium product photography"

    donts = brand.get("donts") or []
    donts_clause = (
        f" Respect these brand don'ts: {'; '.join(donts)}."
        if donts else ""
    )

    campaign_concept = (
        f"Hero shot of the new look for {brand_name}, "
        f"styled in a {voice.lower()} mood."
    )

    # NOTE on colors: we deliberately omit brand hex codes from the
    # brief because Nano Banana treats any color words as cues to
    # restyle the OUTFIT, not just the background. The user's real
    # brand colors get layered on at the Google Ads overlay stage
    # (headline text, CTA button), not in the image itself. Keeping
    # the brief background-only and tone-only is what stops "build
    # ads" from secretly recoloring the garment.
    brief = (
        "Use the attached VTO photograph of the person wearing the "
        "product as the hero, occupying about 60-70 percent of the "
        "frame on the right side of the 1200 by 628 canvas. "
        f"Compose the background to evoke {imagery}, with soft directional "
        "lighting that flatters the subject and a clean depth-of-field "
        "fall-off behind them. "
        "Keep the background tones NEUTRAL and complementary — do NOT "
        "introduce saturated color casts that would tint the person or "
        "the outfit. "
        f"The overall tone should feel {voice.lower()}, on-brand for "
        f"{brand_name}.{donts_clause} "
        "Leave the left side clear for headline text — do NOT render "
        "any text, logo, or wordmark anywhere in the image."
    )

    plan = {
        "vto_artifact_filename": vto,
        "campaign_concept": campaign_concept,
        "briefs": {
            "marketing_image_landscape": brief,
        },
    }
    state["creative_plan"] = plan
    return plan


async def auto_build_creative_plan(
    callback_context: CallbackContext,
    llm_request,  # google.adk.models.LlmRequest, typed positionally
) -> Optional[LlmResponse]:
    """Deterministically build the creative plan and skip the LLM.

    The LLM has repeatedly failed to call `save_creative_plan` reliably
    in this step — sometimes returning empty content, sometimes emitting
    code-execution syntax that ADK rejects. Since the plan is a pure
    function of the brand profile and the hero image (both already in
    state or recoverable from artifacts), we bypass the LLM and build
    the plan in Python.

    The callback writes `state['creative_plan']` and returns a synthetic
    LlmResponse with a single-line confirmation. ADK records this as the
    agent's response and the SequentialAgent moves to the next step.

    If the hero image is missing, we DO NOT fall through to the LLM —
    the LLM has no way to produce a hero out of thin air, and letting
    it run typically results in either a fabricated empty plan or a
    function-call error that leaves state['creative_plan'] unset. The
    asset_generator then fails with "No creative_plan in session state."
    Instead, we return a synthetic LlmResponse with a user-facing text
    that explains exactly what to do.

    Resilience: the plan builder resolves the hero image via the same
    chain as the image-producing tools (vto_image_filename →
    latest_hero_image → newest edit_result_* → newest vto_result_*),
    so a flaky brand_parser run that skipped find_vto_image doesn't
    block the whole campaign.
    """
    plan = await _build_creative_plan_from_state(callback_context)
    if plan is None:
        # No hero image available anywhere in the session. Surface a
        # clear error so the user knows what to do. Logging the same
        # message helps confirm in the deployed-container logs that
        # this branch fired.
        msg = (
            "Cannot build the campaign — no virtual try-on image was "
            "found in this session. Please upload a person photo and a "
            "product photo, ask the agent to try them on, and then ask "
            "for the campaign."
        )
        print(f"[marketing_campaign_agent] auto_build_creative_plan: "
              f"no hero image. {msg}")
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=msg)],
            )
        )

    # Confirm plan was actually written. Belt-and-suspenders log line
    # so future "empty plan" reports are easier to diagnose.
    in_state = callback_context.state.get("creative_plan")
    print(f"[marketing_campaign_agent] auto_build_creative_plan: "
          f"plan built (has_briefs={bool(plan.get('briefs'))}, "
          f"in_state={in_state is not None}, "
          f"vto={plan.get('vto_artifact_filename')!r})")

    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(
                text=f"Concept: {plan['campaign_concept']}"
            )],
        )
    )


creative_director_agent = Agent(
    name="creative_director",
    model="gemini-2.5-flash",
    description=(
        "Plans one Google Responsive Display landscape marketing image "
        "creative brief aligned to the brand guidelines."
    ),
    instruction="""\
You are a senior creative director. You produce ONE function call.

ABSOLUTE RULES:
- Use a FUNCTION CALL only. Never write code. Never emit a code block.
  Never use Python syntax like print(), default_api, triple quotes, or
  raw string prefixes. Do not wrap arguments in ''' or \"\"\".
- Pass each argument as a plain string. Newlines are fine inside the
  string value, but the string itself is a single argument — not a
  multi-line Python literal.
- Do not invent a vto_artifact_filename argument. The tool does not
  accept one; it reads the VTO filename from session state.

CONTEXT YOU CAN READ:
- Session state key brand holds the structured brand profile (colors
  by hex, voice_and_tone, imagery_style, donts).
- The VTO hero image filename is already in state under vto_image_filename.

STEP 1 — Call save_creative_plan with exactly two named arguments:

    campaign_concept   — one short sentence describing the overall
                         campaign idea.
    marketing_image_landscape  — a 3-5 sentence creative brief written
                         as plain prose.

CONTENT OF THE marketing_image_landscape BRIEF:

  Sentence 1 (verbatim, with one bracketed choice filled in): Use the
  attached VTO photograph of the person wearing the product as the
  hero, occupying about 60-70 percent of the frame on the [left or
  right] side of the 1200 by 628 canvas.

  Sentences 2-4: describe the background mood, lighting, and brand
  colors by hex from brand.primary_colors and brand.secondary_colors.
  Brand colors appear only in the background area, never over the
  hero subject. Match brand.voice_and_tone and brand.imagery_style.
  Respect brand.donts.

  Sentence 5: Leave the opposite side clear for headline text — do
  NOT render any text, logo, or wordmark anywhere in the image.

DO NOT write a brief that asks for a logo, wordmark, brand mark, or a
text-only composition. The hero is the person-in-product photograph.

STEP 2 — After save_creative_plan returns, send ONE short text message:
a single line confirming the campaign concept. Nothing else.
""",
    tools=[save_creative_plan],
    # Deterministic plan builder runs before the LLM and (if state has
    # what it needs) returns the plan directly, skipping the LLM call
    # entirely. Falls through to the LLM only if prerequisites are
    # missing — in which case the instruction below kicks in as a
    # fallback.
    before_model_callback=auto_build_creative_plan,
    output_key="creative_plan_ack",
)


async def auto_call_generate_ad_creative(
    callback_context: CallbackContext,
    llm_request,  # google.adk.models.LlmRequest, typed positionally
) -> Optional[LlmResponse]:
    """Force the asset_generator to call generate_ad_creative.

    Same rationale as auto_build_creative_plan: the LLM has been
    unreliable about emitting clean function calls here. Since the
    only legitimate first action for this agent is to call
    generate_ad_creative with the brief from state['creative_plan'],
    we synthesize that call deterministically.

    We only intercept the FIRST model call (when there's no tool
    response yet in the request). Subsequent model calls — the ones
    that follow the tool's return and produce the user-facing markdown
    reply — are left to the LLM, since formatting the final message
    with image embeds is exactly what the LLM is good at.

    Resilience: if creative_plan somehow isn't in state by the time
    this runs (callback ordering quirk, ADK version difference, state
    delta not committed between sub-agents, etc.), we rebuild it on
    the spot from the same primitives. The plan is a pure function of
    state — there's no reason a downstream callback shouldn't be able
    to reconstruct it.
    """
    # Inspect contents: if the most recent part is a function_response,
    # the LLM is being called *after* a tool ran. Let it through so it
    # can build the final reply.
    contents = getattr(llm_request, "contents", None) or []
    for content in reversed(contents):
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "function_response", None):
                return None  # post-tool turn → let LLM compose the reply
            if getattr(part, "function_call", None):
                return None  # already mid-flight
        break  # only check the most recent content

    plan = callback_context.state.get("creative_plan")
    if not isinstance(plan, dict):
        # Plan not in state — rebuild it on the spot. This handles the
        # case where auto_build_creative_plan ran but its state-write
        # didn't propagate (e.g., the SequentialAgent committed the
        # delta between sub-agents in an unexpected way), or didn't
        # run at all due to a callback-dispatch quirk.
        try:
            plan = await _build_creative_plan_from_state(callback_context)
        except Exception as e:
            print(f"[marketing_campaign_agent] "
                  f"auto_call_generate_ad_creative: rebuild failed: "
                  f"{type(e).__name__}: {e}")
            plan = None
        if not isinstance(plan, dict):
            print("[marketing_campaign_agent] "
                  "auto_call_generate_ad_creative: no creative_plan "
                  "available and rebuild produced none — falling "
                  "through to LLM.")
            return None
        print(f"[marketing_campaign_agent] "
              f"auto_call_generate_ad_creative: rebuilt creative_plan "
              f"on the fly (vto={plan.get('vto_artifact_filename')!r})")

    briefs = plan.get("briefs") or {}
    brief = briefs.get("marketing_image_landscape")
    if not brief:
        return None

    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(
                function_call=genai_types.FunctionCall(
                    name="generate_ad_creative",
                    args={
                        "asset_key": "marketing_image_landscape",
                        "creative_brief": brief,
                    },
                )
            )],
        )
    )


asset_generator_agent = Agent(
    name="asset_generator",
    model="gemini-2.5-flash",
    description=(
        "Executes the creative plan by calling the image-generation tool "
        "to produce the landscape Google Display marketing image."
    ),
    instruction="""\
You produce ONE ad creative by calling tools.

ABSOLUTE RULES:
- Use FUNCTION CALLS only. Never write code. Never emit a code block.
  Never use Python syntax like print(), default_api, triple quotes, or
  raw string prefixes. Do not wrap arguments in ''' or \"\"\".
- Pass each argument as a plain string. Newlines inside a string value
  are fine, but the string itself is a single argument — not a
  multi-line Python literal.
- Do all three steps in this single turn; do not pause for the user
  between steps.

STEP 1 — Get the plan.
  Call: get_creative_plan()
  This returns: vto_artifact_filename, campaign_concept, and briefs
  (a dict with one string key: marketing_image_landscape).
  If status is "error", skip to step 3 and report it instead of step 2.

STEP 2 — Generate the landscape marketing image (1.91:1, 1200x628).
  Call: generate_ad_creative(
            asset_key="marketing_image_landscape",
            creative_brief=<briefs.marketing_image_landscape from step 1>)
  Do NOT pass vto_artifact_filename — the tool reads it from session
  state automatically.
  Save the image_markdown field from the result for step 3 (it contains
  the inline image embed already formatted).

STEP 3 — Reply to the user with ONE text message.

On success, the message is exactly:

Concept: <campaign_concept>

<paste image_markdown from step 2 here verbatim, do not modify it>

Asset meets Google Responsive Display requirements (1.91:1, 1200x628,
PNG, <5MB). Upload directly to Google Ads.

On error, replace the image_markdown line with:
    marketing_image_landscape — failed: <message from the tool result>
""",
    tools=[get_creative_plan, generate_ad_creative],
    # Skip the unreliable "decide which tool to call" model step and
    # synthesize the generate_ad_creative call directly from state. The
    # LLM is still used for the FINAL turn (the one that produces the
    # user-facing markdown reply after the tool returns).
    before_model_callback=auto_call_generate_ad_creative,
)


social_media_agent = Agent(
    name="social_media_publisher",
    model="gemini-2.5-flash",
    description=(
        "Creates ready-to-publish Instagram and TikTok posts from the VTO "
        "image: native-aspect images plus platform-specific captions and "
        "hashtags."
    ),
    instruction="""\
You produce three social media posts from the VTO hero image:
  1. Instagram feed (4:5, 1080x1350)
  2. Instagram Reels/Story cover (9:16, 1080x1920)
  3. TikTok vertical (9:16, 1080x1920)

ABSOLUTE RULES:
- Use FUNCTION CALLS only. Never write code. Never emit a code block.
  Never use Python syntax like print(), default_api, triple quotes, or
  raw string prefixes. Do not wrap arguments in ''' or \"\"\".
- Pass each argument as a plain string (or for hashtags, a plain list
  of strings). Newlines are fine inside a string value, but the string
  is a single argument — not a multi-line Python literal.
- Make ALL FOUR steps in this single turn. Do not pause for the user.
- If a tool call returns status="error", STILL call the remaining ones.
- The visual_brief must instruct the model to use the attached VTO
  photo as the HERO. Do not write briefs that ask for new scenes —
  the photo IS the scene.

CONTEXT YOU CAN READ:
- state['vto_image_filename'] — the hero photo (already located).
- state['brand'] — brand profile (voice_and_tone, colors, donts).
- state['creative_plan'] — the ads-side plan (you can borrow the
  campaign_concept for tonal consistency, but social copy should be
  more casual than the ads).

STEP 1 — Instagram feed post.
  Call: generate_social_post(
            platform_key="instagram_feed",
            visual_brief=<3-4 sentence visual brief, see rules>,
            caption=<Instagram caption, see voice rules>,
            hashtags=<5-10 hashtags as a list of strings, no leading #>)

STEP 2 — Instagram Reels/Story cover.
  Call: generate_social_post(
            platform_key="instagram_reels_story",
            visual_brief=<recompose for 9:16 vertical, hero centered, top
                          and bottom 250px kept free of important detail>,
            caption=<short Reels caption, hook first, 1-3 sentences>,
            hashtags=<5-8 hashtags, lean toward trending/Reels-friendly>)

STEP 3 — TikTok vertical.
  Call: generate_social_post(
            platform_key="tiktok_vertical",
            visual_brief=<9:16 vertical, person fills frame, more candid /
                          less studio-perfect than the Instagram versions,
                          plenty of headroom for TikTok's bottom UI
                          (~340px) and right-rail (~150px)>,
            caption=<TikTok caption: hook in first 6 words, conversational,
                     can reference trending sound or format>,
            hashtags=<5-10 hashtags, mix of broad (#fashion, #ootd) and
                      niche/trending. Always include #fyp and #foryou
                      for TikTok>)

VISUAL BRIEF RULES (apply to all three calls):
- The brief always starts with: "Use the attached VTO photograph as the
  hero, [framing instructions for this aspect ratio]."
- Describe the background as native-feeling for the platform (Instagram:
  curated, magazine-lite; TikTok: candid, room-lit, less staged).
- Reference brand colors by hex only for ACCENTS in the background.
- Closing line: "Do NOT render text, logos, or watermarks anywhere in
  the image — the platform overlays captions and UI at display time."

CAPTION/COPY RULES:
- Match brand.voice_and_tone. If the brand voice is "playful and bold",
  the captions sound playful and bold.
- Instagram feed captions can be 2-4 sentences with light line breaks.
- Reels/TikTok captions should be 1-3 sentences with a strong hook in
  the first 6 words (TikTok cuts off after that in many UI states).
- Never use generic stock copy like "Check it out!" or "Link in bio."
  Give the user something they'd actually post.
- No celebrity names, no trademarked phrases, no copyrighted lyrics.

STEP 4 — Reply to the user with ONE text message in EXACTLY this
format. Each section is the `post_text_markdown` value returned by the
matching tool call. Paste each markdown block VERBATIM — do not
summarise, do not replace the caption with a filename, do not mention
".txt" anywhere. The user needs to see the actual caption text.

Social posts ready to publish:

<post_text_markdown value from step 1, pasted verbatim>

---

<post_text_markdown value from step 2, pasted verbatim>

---

<post_text_markdown value from step 3, pasted verbatim>

For any post that failed, replace its section with a single line:
    <platform_key> — failed: <message from the tool result>
""",
    tools=[generate_social_post],
)


# ===========================================================================
# Pipeline (sequential workflow)
# ===========================================================================

marketing_campaign_agent = SequentialAgent(
    name="marketing_campaign_agent",
    description=(
        "Generates a Google Ads Responsive Display landscape marketing "
        "image (1.91:1, 1200x628) from a virtual-try-on image and a "
        "brand guidelines PDF. Three steps: locate VTO + parse brand PDF, "
        "plan the ad creative, generate the asset via Nano Banana. "
        "Does NOT produce social media posts — use "
        "social_media_publisher_agent for that."
    ),
    sub_agents=[
        brand_parser_agent,
        creative_director_agent,
        asset_generator_agent,
    ],
)


social_media_publisher_agent = SequentialAgent(
    name="social_media_publisher_agent",
    description=(
        "Generates Instagram and TikTok social posts (image + caption + "
        "hashtags) from a virtual-try-on image and a brand guidelines "
        "PDF. Two steps: locate VTO + parse brand PDF, generate three "
        "platform-tuned posts (Instagram feed 4:5, Instagram "
        "Reels/Story 9:16, TikTok 9:16)."
    ),
    sub_agents=[
        brand_parser_agent_for_social,
        social_media_agent,
    ],
)


# Exposed as `root_agent` so this package can also run standalone via
# `adk web` / `adk run marketing_campaign_agent`. Default: ads pipeline.
root_agent = marketing_campaign_agent