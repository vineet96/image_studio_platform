#!/usr/bin/env python3
"""
Deploy `image_studio_root_agent` to Vertex AI Agent Engine.

The root agent depends on three sibling packages:
    vto_agent/                           ← virtual_try_on + edit_image tools
    marketing_campaign_agent/            ← campaign pipeline (SequentialAgent)
    image_studio_root_agent/             ← root agent (this is what's deployed)

`extra_packages` bundles all three OUTER package directories so the
deployed container has them on PYTHONPATH and the cloudpickled agent
can resolve cross-package imports at runtime.

Usage
-----
    # First time (create):
    python deploy/deploy.py --project YOUR_PROJECT --bucket gs://YOUR_BUCKET

    # Update existing deployment:
    python deploy/deploy.py --project YOUR_PROJECT --bucket gs://YOUR_BUCKET \
                            --resource-id 1234567890123456789

    # Delete:
    python deploy/deploy.py --project YOUR_PROJECT --delete \
                            --resource-id 1234567890123456789

    # Plan only (no API calls):
    python deploy/deploy.py --project YOUR_PROJECT --bucket gs://YOUR_BUCKET \
                            --dry-run

Flags
-----
--project       GCP project ID. Falls back to $GOOGLE_CLOUD_PROJECT.
--location      Region. Default us-central1.
--bucket        GCS staging bucket gs://... (required for create/update).
--resource-id   Existing reasoning-engine ID → update; else create.
--display-name  Display name in the Agent Engine UI.
--delete        Delete the engine identified by --resource-id and exit.
--dry-run       Print plan, no API calls.
--smoke-test    After deploy, send a hello query to verify it's reachable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — make sibling packages importable BEFORE we try to import
# the root agent (which transitively imports from vto_agent.agent and
# marketing_campaign_agent.agent).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("vto_agent", "marketing_campaign_agent", "image_studio_root_agent"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_sdk():
    """Import the Vertex SDK only after arg parsing so --help works without
    the dependency installed."""
    import vertexai
    from vertexai import agent_engines
    from vertexai.preview.reasoning_engines import AdkApp
    return vertexai, agent_engines, AdkApp


# ---------------------------------------------------------------------------
# Deployment configuration
# ---------------------------------------------------------------------------

# Pinned runtime dependencies installed inside the Agent Engine container.
# Agent Engine reads ONLY this list — your local requirements.txt is
# ignored at deploy time. Add new deps here, not in pip install commands.
REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,adk]>=1.112",
    "google-adk>=1.15.0",
    "google-genai>=1.0.0",
    "google-cloud-storage>=2.10.0",
    "Pillow>=10.0.0",
]

# Local source bundled into the deployment archive.
#
# The Vertex AI SDK's tar packager calls `tar.add(path)` with no
# `arcname`, so each entry in the resulting tarball preserves the
# path you give it verbatim. If you pass an absolute path like
# `/home/user/.../marketing_campaign_agent`, the tar entry becomes
# `home/user/.../marketing_campaign_agent/...`, which extracts on the
# runtime container into a deeply nested directory that's not on
# `sys.path` — so `import marketing_campaign_agent.agent` fails with
# `No module named 'marketing_campaign_agent'`.
#
# We work around this by staging the three inner package directories
# (each containing `__init__.py` + `agent.py`) into a flat temp dir,
# changing to that dir, and listing each package by its bare basename.
# Then the tar entries are simply `vto_agent/`, `marketing_campaign_agent/`,
# and `image_studio_root_agent/`, which extract directly into the
# container's working dir (on `sys.path`).
#
# See _stage_extra_packages() below for implementation.
_INNER_PACKAGE_PATHS = [
    REPO_ROOT / "vto_agent" / "vto_agent",
    REPO_ROOT / "marketing_campaign_agent" / "marketing_campaign_agent",
    REPO_ROOT / "image_studio_root_agent" / "image_studio_root_agent",
]


def _stage_extra_packages() -> tuple[Path, list[str]]:
    """Stage the inner package directories into a flat temp dir.

    Returns:
        (staging_dir, [basename, ...]) — the caller should chdir to
        staging_dir before calling agent_engines.create/update so that
        the SDK's `tarfile.add(basename)` produces flat tar entries.
    """
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="image_studio_deploy_"))
    basenames: list[str] = []
    for src in _INNER_PACKAGE_PATHS:
        if not (src / "agent.py").is_file():
            raise SystemExit(
                f"Inner package missing agent.py: {src}. Repo layout broken."
            )
        dst = staging / src.name
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        basenames.append(src.name)
    return staging, basenames


def build_env_vars(
    brand_pdf_bucket: str = "your GCS bucket",
    brand_pdf_prefix: str = "brands/",
) -> dict[str, str]:
    """Env vars baked into the deployed Agent Engine runtime.

    IMPORTANT — Agent Engine rejects deployments that try to set
    "reserved" environment variable names because it injects them
    automatically inside the container. Per Google's docs the reserved
    set is:
        GOOGLE_CLOUD_PROJECT
        GOOGLE_CLOUD_QUOTA_PROJECT
        GOOGLE_CLOUD_LOCATION
        PORT
        K_SERVICE
        K_REVISION
        K_CONFIGURATION
        GOOGLE_APPLICATION_CREDENTIALS
    Passing any of these in env_vars causes a FailedPrecondition error
    at deploy time:
        "Environment variable name 'X' is reserved.
         Please rename the variable in spec.deployment_spec.env."

    So we explicitly do NOT set GOOGLE_CLOUD_PROJECT or
    GOOGLE_CLOUD_LOCATION here — Agent Engine provides them.

    GOOGLE_GENAI_USE_VERTEXAI is NOT reserved, but we don't need it
    either: our agent code now constructs `genai.Client(vertexai=True,
    project=..., location=...)` explicitly, reading from the
    auto-injected GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION. The flag
    becomes redundant.

    Do NOT put secrets here in plaintext for production — use Secret
    Manager + reference secret resource names instead.
    """
    env = {
        # Writable scratch dir inside the container (Pillow falls back here
        # in addition to saving each image as an artifact).
        "CAMPAIGN_OUTPUT_DIR": "/tmp/campaign_outputs",
    }
    if brand_pdf_bucket:
        env["BRAND_PDF_BUCKET"] = brand_pdf_bucket
    if brand_pdf_prefix:
        env["BRAND_PDF_PREFIX"] = brand_pdf_prefix
    return env


DEFAULT_DESCRIPTION = (
    "Image Studio agent: virtual try-on (Google virtual-try-on-001) + "
    "generative image editing (Nano Banana) + marketing campaign "
    "generation. From a VTO image and a brand guidelines PDF, produces "
    "one Google Ads Responsive Display landscape image (1.91:1, "
    "1200x628), one Instagram feed post (4:5, 1080x1350), one Instagram "
    "Reels/Story cover (9:16, 1080x1920), and one TikTok vertical post "
    "(9:16, 1080x1920), with platform-tuned captions and hashtags."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",
                   default=os.getenv("GOOGLE_CLOUD_PROJECT"),
                   help="GCP project ID (or $GOOGLE_CLOUD_PROJECT).")
    p.add_argument("--location",
                   default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                   help="Region. Default: us-central1.")
    p.add_argument("--bucket",
                   default=os.getenv("AGENT_ENGINE_STAGING_BUCKET"),
                   help="GCS staging bucket gs://... "
                        "(or $AGENT_ENGINE_STAGING_BUCKET).")
    p.add_argument("--resource-id",
                   help="Existing reasoning-engine ID. If set → UPDATE; "
                        "else CREATE.")
    p.add_argument("--display-name",
                   default="Image Studio (VTO + Campaign + Social)",
                   help="Display name in the Agent Engine UI.")
    p.add_argument("--description", default=DEFAULT_DESCRIPTION,
                   help="Description shown in the Agent Engine UI.")
    p.add_argument("--delete", action="store_true",
                   help="Delete the engine identified by --resource-id "
                        "and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan, make no API calls.")
    p.add_argument("--smoke-test", action="store_true",
                   help="After deploy, send a hello query to verify "
                        "the engine is reachable.")
    p.add_argument("--brand-pdf-bucket",
                   default=os.getenv("BRAND_PDF_BUCKET", ""),
                   help="GCS bucket containing brand-guidelines PDFs "
                        "(no gs:// prefix). When set, the deployed "
                        "agent auto-pulls the newest .pdf from this "
                        "bucket and uses it as the brand guidelines. "
                        "Falls back to $BRAND_PDF_BUCKET.")
    p.add_argument("--brand-pdf-prefix",
                   default=os.getenv("BRAND_PDF_PREFIX", ""),
                   help="Optional path prefix inside the brand PDF "
                        "bucket (e.g. 'brands/'). Falls back to "
                        "$BRAND_PDF_PREFIX.")
    return p.parse_args()


def require(cond: bool, msg: str) -> None:
    if not cond:
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def do_delete(args, agent_engines) -> None:
    full_name = (f"projects/{args.project}/locations/{args.location}"
                 f"/reasoningEngines/{args.resource_id}")
    print(f"Deleting Agent Engine {full_name} ...")
    agent_engines.get(full_name).delete(force=True)
    print("Deleted.")


def _strip_reserved_env(env: dict[str, str]) -> dict[str, str]:
    """Defense-in-depth: remove any Agent Engine reserved env var names.

    If someone ever puts one of these back into build_env_vars, the
    deploy still succeeds and we print a warning. Without this, the
    create/update call fails with FailedPrecondition: "Environment
    variable name 'X' is reserved."
    """
    reserved = {
        "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_QUOTA_PROJECT",
        "GOOGLE_CLOUD_LOCATION", "PORT", "K_SERVICE", "K_REVISION",
        "K_CONFIGURATION", "GOOGLE_APPLICATION_CREDENTIALS",
    }
    cleaned = {k: v for k, v in env.items() if k not in reserved}
    dropped = sorted(set(env) - set(cleaned))
    if dropped:
        print(f"⚠️  Dropping reserved env var names from deployment: "
              f"{dropped}. Agent Engine injects these automatically.")
    return cleaned


def do_deploy(args, agent_engines, AdkApp) -> None:
    # Import the agent inside the function so import-time errors surface
    # clearly here, not at the top of the script. (Agents construct genai
    # clients on import, which can fail with confusing errors if env vars
    # are wrong; we want those errors to land here.)
    from image_studio_root_agent.agent import root_agent

    print("Wrapping root_agent in AdkApp ...")
    app = AdkApp(agent=root_agent, enable_tracing=True)

    # Stage the inner packages into a flat temp dir, then we'll chdir
    # there before invoking the SDK. The SDK tars by passing each path
    # verbatim to tarfile.add() with no arcname, so we need the paths
    # to be bare basenames relative to CWD — otherwise the deployed
    # archive contains a deeply-nested tree that's not importable.
    print("Staging inner packages for deployment archive...")
    staging_dir, extra_basenames = _stage_extra_packages()
    print(f"  staging dir : {staging_dir}")
    print(f"  basenames   : {extra_basenames}")

    common_kwargs = dict(
        agent_engine=app,
        requirements=REQUIREMENTS,
        extra_packages=extra_basenames,
        env_vars=_strip_reserved_env(build_env_vars(
            args.brand_pdf_bucket,
            args.brand_pdf_prefix,
        )),
        display_name=args.display_name,
        description=args.description,
    )

    print("\nDeployment plan")
    print("---------------")
    print(f"  Project        : {args.project}")
    print(f"  Location       : {args.location}")
    print(f"  Staging bucket : {args.bucket}")
    print(f"  Display name   : {args.display_name}")
    if args.resource_id:
        print(f"  Mode           : UPDATE (resource_id={args.resource_id})")
    else:
        print(f"  Mode           : CREATE")
    print(f"  Requirements   :")
    for r in REQUIREMENTS:
        print(f"    - {r}")
    print(f"  Extra packages : (staged, then tar'd as flat basenames)")
    for base in extra_basenames:
        pkg = staging_dir / base
        ok = "✓" if (pkg / "agent.py").is_file() else "MISSING"
        print(f"    - {base}/  [{ok}]  (from {pkg})")
    print(f"  Env vars       : {list(common_kwargs['env_vars'].keys())}")
    if args.brand_pdf_bucket:
        prefix_suffix = f"/{args.brand_pdf_prefix}" if args.brand_pdf_prefix else ""
        print(f"  Brand PDFs     : gs://{args.brand_pdf_bucket}{prefix_suffix} "
              f"(GCS-backed)")
    else:
        print(f"  Brand PDFs     : chat attachments only "
              f"(set --brand-pdf-bucket to use GCS)")
    print(f"  Outputs        : 1 Google Ad (1.91:1) + 3 social posts "
          f"(IG feed 4:5, IG Reels/Story 9:16, TikTok 9:16)")

    if args.dry_run:
        print("\n--dry-run set — exiting before API call.")
        return

    # Pre-flight: confirm each staged package actually contains agent.py
    # and __init__.py at its root.
    for base in extra_basenames:
        pkg_path = staging_dir / base
        require(
            (pkg_path / "agent.py").is_file(),
            f"staged package {pkg_path} missing agent.py.",
        )
        require(
            (pkg_path / "__init__.py").is_file(),
            f"staged package {pkg_path} missing __init__.py.",
        )

    t0 = time.time()
    # chdir to staging so the SDK's tar.add(basename) records flat
    # entries. We restore cwd afterward in a finally block.
    original_cwd = os.getcwd()
    os.chdir(staging_dir)
    try:
        if args.resource_id:
            full_name = (f"projects/{args.project}/locations/{args.location}"
                         f"/reasoningEngines/{args.resource_id}")
            print(f"\nUpdating {full_name} ...")
            remote = agent_engines.get(full_name).update(**common_kwargs)
        else:
            print("\nCreating new Agent Engine "
                  "(this typically takes 5-10 minutes) ...")
            remote = agent_engines.create(**common_kwargs)
    finally:
        os.chdir(original_cwd)

    dt = time.time() - t0
    short_id = remote.resource_name.rsplit("/", 1)[-1]
    print(f"\n✅ Done in {dt:.0f}s")
    print(f"Resource name : {remote.resource_name}")
    print(f"Resource ID   : {short_id}")
    print(f"Console       : "
          f"https://console.cloud.google.com/vertex-ai/agents/"
          f"agent-engines/locations/{args.location}/agent-engines/"
          f"{short_id}?project={args.project}")
    print(f"\nReuse in code:")
    print(f"    from vertexai import agent_engines")
    print(f"    engine = agent_engines.get('{remote.resource_name}')")
    print(f"\nFor subsequent UPDATE deploys, pass:")
    print(f"    --resource-id {short_id}")

    if args.smoke_test:
        print("\nRunning smoke test ...")
        try:
            for ev in remote.stream_query(
                user_id="deploy-smoke-test",
                message="Hello — please confirm you are deployed and "
                        "list your capabilities briefly.",
            ):
                # Trim noisy fields for readability.
                print(" •", str(ev)[:300])
            print("Smoke test OK.")
        except Exception as e:
            print(f"⚠️  Smoke test failed: {type(e).__name__}: {e}")
            print("   The engine is likely deployed but unreachable, "
                  "or the agent failed on first invocation. Check the "
                  "console URL above for runtime logs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    require(args.project, "Set --project or $GOOGLE_CLOUD_PROJECT.")
    if args.delete:
        require(args.resource_id, "--delete requires --resource-id.")
    else:
        require(args.bucket and args.bucket.startswith("gs://"),
                "Set --bucket gs://your-bucket "
                "(or $AGENT_ENGINE_STAGING_BUCKET).")

    # Export Vertex AI env vars BEFORE importing the agent. The agent
    # modules construct a genai.Client at import time. With this set
    # here, that constructor sees vertex mode and doesn't try the
    # Gemini Developer API key path (which would fail with "No API key
    # was provided" if the operator's shell hasn't been pre-configured).
    # The same vars also get baked into the deployed engine via
    # build_env_vars() — this is just for the local import step.
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    if getattr(args, "brand_pdf_bucket", ""):
        os.environ["BRAND_PDF_BUCKET"] = args.brand_pdf_bucket
    if getattr(args, "brand_pdf_prefix", ""):
        os.environ["BRAND_PDF_PREFIX"] = args.brand_pdf_prefix

    vertexai, agent_engines, AdkApp = _load_sdk()

    print(f"Init Vertex AI SDK (project={args.project}, "
          f"location={args.location}) ...")
    vertexai.init(
        project=args.project,
        location=args.location,
        staging_bucket=args.bucket if not args.delete else None,
    )

    if args.delete:
        do_delete(args, agent_engines)
    else:
        do_deploy(args, agent_engines, AdkApp)


if __name__ == "__main__":
    main()
