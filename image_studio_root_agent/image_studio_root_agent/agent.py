"""
Image Studio root agent.

Takes the existing VTO agent's tools (`virtual_try_on`, `edit_image`) and
adds the `marketing_campaign_agent` SequentialAgent as a sub-agent so the
user can:

  1. Run virtual try-on or image editing as before.
  2. Say "make ads from this", upload a brand PDF, and the root agent
     transfers to the campaign sub-agent — deterministically, via a
     before_model_callback that doesn't depend on the LLM cooperating.

The whole tree is wrapped in a single ADK App with the
SaveFilesAsArtifactsPlugin so uploads (person photo, product photo,
brand PDF) all auto-save as artifacts that any agent in the tree can
load via `tool_context.load_artifact(...)`.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap. See vto_agent/__init__.py for rationale.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sibling in ("vto_agent", "marketing_campaign_agent"):
    _p = str(_REPO_ROOT / _sibling)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from google.adk.agents import Agent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.apps import App
    from google.adk.models import LlmResponse
    from google.adk.plugins.save_files_as_artifacts_plugin import (
        SaveFilesAsArtifactsPlugin,
    )
    from google.genai import types as genai_types

    from vto_agent.agent import virtual_try_on, edit_image, reset_session
    from marketing_campaign_agent.agent import (
        marketing_campaign_agent,
        social_media_publisher_agent,
    )
except Exception:
    traceback.print_exc()
    raise


# ---------------------------------------------------------------------------
# Deterministic transfer backstop.
#
# The LLM has repeatedly been observed to paraphrase the transfer ("I'll
# hand this to the campaign agent...") instead of actually emitting a
# `transfer_to_agent` function call. No amount of prompt prohibition has
# eliminated this. So we don't depend on the LLM for this routing
# decision — we intercept before the model runs and force a transfer
# whenever the user message matches campaign intent.
#
# ADK's before_model_callback returns:
#   None         → let the LLM run normally
#   LlmResponse  → skip the LLM, use this response as if the model
#                  produced it. By returning an LlmResponse containing a
#                  `transfer_to_agent` function-call part, we force ADK's
#                  agent-transfer machinery without the LLM's judgement.
# ---------------------------------------------------------------------------

# Two intent regexes — one per target sub-agent. They can both match the
# same message ("build a campaign with ads and social posts"), in which
# case we transfer to the ads agent first; once it finishes and control
# returns to the root, the second user-turn-derived callback fires and
# routes to social.
#
# Both regexes are intentionally broad. Users phrase these requests in
# many ways ("create posts", "make a banner", "design a promo"), and
# false positives are cheap — the destination pipeline reports clean
# errors if prerequisites are missing.

# Ads-only intent.
_AD_INTENT = re.compile(
    r"\b("
    r"campaign|"
    r"ads?|advert|advertising|"
    r"google\s*ads?|"
    r"creatives?|"
    r"marketing\s*(assets?|materials?|images?|creatives?)?|"
    r"banners?|posters?|"
    r"promos?|promotional|promotions?|"
    r"display\s*ads?|"
    r"brand\s*guidelines"
    r")\b",
    re.IGNORECASE,
)

# Social-only intent. Adding "post(s)" / "content" as standalone triggers
# means a user can say "create posts" or "create some content based on
# the guidelines" and it routes correctly. The ad pipeline doesn't claim
# those words.
_SOCIAL_INTENT = re.compile(
    r"\b("
    r"social\s*(media|posts?|content)?|"
    r"instagram|ig|"
    r"reels?|stories|story|"
    r"tiktok|tik\s*tok|"
    r"posts?|"                  # bare "post(s)" — broad on purpose
    r"content"                  # bare "content" — same
    r")\b",
    re.IGNORECASE,
)


def _latest_user_text(llm_request) -> str:
    """Pull the most recent user-authored text from the LLM request.

    `llm_request.contents` is a list of `genai.types.Content`; we walk
    from the end and return the first user-role text part. This is the
    documented callback-input shape.
    """
    contents = getattr(llm_request, "contents", None) or []
    for content in reversed(contents):
        if getattr(content, "role", None) != "user":
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                return text
    return ""


def maybe_force_campaign_transfer(
    callback_context: CallbackContext,
    llm_request,  # google.adk.models.LlmRequest — typed positionally
) -> Optional[LlmResponse]:
    """Route campaign / social intents to the right sub-agent.

    Intent detection is two separate regexes:
      _AD_INTENT     matches "campaign", "ads", "marketing assets", etc.
      _SOCIAL_INTENT matches "social", "Instagram", "TikTok", "Reels", etc.

    Possible cases:
      - Only social intent → transfer to social_media_publisher_agent.
      - Only ad intent     → transfer to marketing_campaign_agent.
      - Both intents       → transfer to marketing_campaign_agent first;
        flag social_pending=True on state so a follow-up callback fires
        once control returns to the root (no extra user prompt needed).
      - Neither            → return None, let the LLM handle.

    One-shot guard per user turn prevents re-firing inside the same
    transferred-from-sub-agent loop.
    """
    user_text = _latest_user_text(llm_request)
    state = callback_context.state

    # If we're returning to the root after a prior transfer, check
    # whether we still owe the user a social-pipeline run.
    if state.get("_social_pending"):
        state["_social_pending"] = False
        return _transfer_response("social_media_publisher_agent")

    if not user_text:
        return None

    wants_ads = bool(_AD_INTENT.search(user_text))
    wants_social = bool(_SOCIAL_INTENT.search(user_text))

    if not (wants_ads or wants_social):
        return None

    # Re-fire guard.
    #
    # We want two things at once:
    #
    #   (a) Don't re-fire mid-turn. When a sub-agent yields control
    #       back to the root, the root's LLM callback runs again —
    #       same user message in context, so without a guard the
    #       transfer would fire again forever.
    #
    #   (b) DO re-fire on the next user turn, even if they typed
    #       exactly the same sentence. Previously we deduped on
    #       (user_text, intent_combo, hero_image), which meant a
    #       user who retyped "create social media posts based on
    #       guidelines attached here" after the first attempt
    #       failed would see the agent silently ignore them.
    #
    # The proxy for "same turn vs new turn" is the count of
    # user-authored contents in the LLM request. It increases by 1
    # each time the user sends a new message and stays constant
    # while sub-agents loop control back. Storing the user-turn
    # count at last transfer + requiring strict increase before the
    # next transfer satisfies both goals.
    user_turn_count = sum(
        1 for c in (getattr(llm_request, "contents", None) or [])
        if getattr(c, "role", None) == "user"
    )
    last_turn_count = state.get("_transferred_at_user_turn", -1)
    if user_turn_count <= last_turn_count:
        # Same user turn as the last transfer → swallow.
        return None
    state["_transferred_at_user_turn"] = user_turn_count

    if wants_ads and wants_social:
        # Run ads first, then queue social for the next pass.
        state["_social_pending"] = True
        return _transfer_response("marketing_campaign_agent")
    if wants_ads:
        return _transfer_response("marketing_campaign_agent")
    return _transfer_response("social_media_publisher_agent")


def _transfer_response(agent_name: str) -> LlmResponse:
    """Build the LlmResponse that triggers an agent transfer."""
    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        name="transfer_to_agent",
                        args={"agent_name": agent_name},
                    )
                )
            ],
        )
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

INSTRUCTION = """\
You help users with two capabilities. (A third capability, marketing
campaign generation, is handled automatically by the framework via a
sub-agent — you never need to think about it.)

A) VIRTUAL TRY-ON — show a person wearing a clothing product.
   Use the `virtual_try_on` tool with two uploaded images (person + product).
   If either is missing, ask the user to upload both.

B) IMAGE EDITING (Nano Banana) — apply a generative edit to a single image.
   Use the `edit_image` tool.

   When calling edit_image, ALWAYS pass source_filename="auto" unless
   the user has explicitly named a specific image file. The tool will
   automatically pick the most recent image in the session — typically
   the latest virtual_try_on output, or the latest prior edit, or a
   user upload. You should NOT ask the user "which image?" — the tool
   handles that. The only thing you need from the user is a clear
   edit instruction.

   If the user is vague (e.g. "change the color"), ask a clarifying
   question (e.g. "Which color?") BEFORE calling the tool.

C) RESET / START OVER — call `reset_session` (no arguments) when the
   user says anything like: "start over", "reset", "new campaign",
   "forget what we did", "different person now", or otherwise signals
   they want to wipe the previous run. This clears state cursors so
   the next virtual_try_on starts clean. After calling reset_session,
   briefly tell the user the slate is clear and ask what they'd like
   to do next.

When VTO or edit tools succeed, reply with a SHORT confirmation and embed
the generated image using the `markdown` field returned by the tool.
Paste the markdown verbatim.

Keep replies short. Never echo base64 or raw bytes outside the markdown
field returned by tools.
"""


try:
    root_agent = Agent(
        name="image_studio_root_agent",
        model="gemini-2.5-flash",
        description=(
            "Image studio: virtual try-on, Nano Banana editing, plus "
            "marketing campaign generation (Google Ad + Instagram + "
            "TikTok) routed to a sub-agent."
        ),
        instruction=INSTRUCTION,
        tools=[virtual_try_on, edit_image, reset_session],
        sub_agents=[marketing_campaign_agent, social_media_publisher_agent],
        # The callback runs before each LLM invocation. If it returns an
        # LlmResponse, ADK skips the model and uses that response. This
        # is how we force the campaign transfer deterministically.
        before_model_callback=maybe_force_campaign_transfer,
    )

    # `App.name` must equal the package directory name
    # (`image_studio_root_agent`) so `adk web` / `adk run` find sessions.
    app = App(
        name="image_studio_root_agent",
        root_agent=root_agent,
        plugins=[SaveFilesAsArtifactsPlugin()],
    )
except Exception:
    traceback.print_exc()
    raise