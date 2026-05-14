# Image Studio Platform

A Google ADK multi-agent system for fashion / e-commerce workflows. Three
capabilities, one deployment:

1. **Virtual try-on** — Google `virtual-try-on-001` (person + product → person wearing product).
2. **Generative image editing** — Nano Banana (`gemini-2.5-flash-image`) for color changes, swaps, removals.
3. **Marketing campaign generation** — turns the try-on image plus a brand guidelines PDF into:
   - A Google Ads **Responsive Display landscape marketing image** (1.91:1, 1200×628), on-brand and at exact spec.
   - **Instagram feed post** (4:5, 1080×1350) with caption + hashtags.
   - **Instagram Reels / Story cover** (9:16, 1080×1920) with caption + hashtags.
   - **TikTok vertical post** (9:16, 1080×1920) with caption + hashtags.

## Repository layout

```
image_studio_platform/
├── README.md                                    ← you are here
├── docs/
│   └── architecture.md                          ← diagrams, state flow, Google Ads spec mapping
│
├── vto_agent/                                   ← Agent: VTO + Nano Banana editing (standalone)
│   ├── README.md
│   ├── requirements.txt
│   └── vto_agent/
│       ├── __init__.py
│       ├── agent.py                             ← virtual_try_on + edit_image tools
│       └── .env.example
│
├── marketing_campaign_agent/                    ← Agent: brand-aware ad-creative pipeline (standalone)
│   ├── README.md
│   ├── requirements.txt
│   └── marketing_campaign_agent/
│       ├── __init__.py
│       ├── agent.py                             ← SequentialAgent: brand_parser → creative_director → asset_generator → social_media_publisher
│       └── .env.example
│
├── image_studio_root_agent/                     ← Root agent: VTO tools + campaign sub-agent (deployable)
│   ├── README.md
│   ├── requirements.txt
│   └── image_studio_root_agent/
│       ├── __init__.py
│       ├── agent.py                             ← root_agent (tools=VTO) + sub_agents=[marketing_campaign_agent]
│       └── .env.example
│
└── deploy/                                      ← Vertex AI Agent Engine deployment
    ├── deploy.py                                ← create / update / delete
    └── invoke_remote.py                         ← call the deployed engine
```

The three agent packages are **siblings**, not nested. `image_studio_root_agent`
imports `virtual_try_on` and `edit_image` from `vto_agent.agent`, and
`marketing_campaign_agent` from `marketing_campaign_agent.agent`.

`image_studio_root_agent/agent.py` contains a `sys.path` bootstrap that walks
up to the repo root and adds the two sibling package directories, so local
`adk web` / `adk run` works without setting `PYTHONPATH`. Agent Engine
deployments don't need this — `deploy/deploy.py` bundles all three packages
via `extra_packages` so they end up on `PYTHONPATH` inside the container.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  image_studio_root_agent  (one Agent Engine instance)                    │
│                                                                          │
│  root_agent  (LlmAgent, gemini-2.5-flash)                                │
│  ├── tools:     virtual_try_on, edit_image          ← from vto_agent     │
│  └── sub_agents:                                                         │
│      └── marketing_campaign_agent  (SequentialAgent)                     │
│          ├── brand_parser          (gemini-2.5-flash + PDF reading)      │
│          ├── creative_director     (gemini-2.5-flash)                    │
│          ├── asset_generator       (gemini-2.5-flash + image tools)      │
│          └── social_media_publisher (gemini-2.5-flash + image tools)     │
│                                                                          │
│  Plugin: SaveFilesAsArtifactsPlugin                                      │
│    Every uploaded image / PDF becomes an artifact in the session         │
│    store, accessible to every agent in the tree via tool_context.        │
└──────────────────────────────────────────────────────────────────────────┘
```

**Shared state** is how the pipeline composes — no manual file passing:

- The VTO tool writes its output as artifact `vto_result_xxxx.png`.
- The marketing pipeline's `creative_director` reads that same artifact from
  the session.
- `brand_parser` writes `state['brand']`, `creative_director` writes
  `state['creative_plan']`, `asset_generator` reads both.

## Quick start

```bash
# 0. Prereqs
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com \
    storage.googleapis.com cloudbuild.googleapis.com

# 1. Install
pip install -r image_studio_root_agent/requirements.txt

# 2. Configure
cp image_studio_root_agent/image_studio_root_agent/.env.example \
   image_studio_root_agent/image_studio_root_agent/.env
# edit .env: set GOOGLE_CLOUD_PROJECT

# 3. Run locally — from the repo root
adk web image_studio_root_agent          # browser UI at http://localhost:8000
# or
adk run image_studio_root_agent          # terminal chat

# 4. Deploy to Agent Engine
python deploy/deploy.py --project YOUR_PROJECT --bucket gs://YOUR_STAGING_BUCKET
```

Each standalone agent (`vto_agent/`, `marketing_campaign_agent/`) can also be

## Brand guidelines: GCS or attachment

The campaign and social-media pipelines need a brand-guidelines PDF.
Two ways to provide it:

### Option 1 — GCS bucket (recommended)

Drop one or more `.pdf` files into a Google Cloud Storage bucket. Set
two env vars (in `.env` or via deploy flags) and the agent auto-pulls
the newest PDF in the bucket on every campaign request:

```
BRAND_PDF_BUCKET=my-brand-pdfs        # bucket name, no gs:// prefix
BRAND_PDF_PREFIX=brands/              # optional sub-folder
```

The user just says "build the ads campaign" — no attachment needed.
Multiple PDFs? The newest by GCS `updated` timestamp wins.

The bucket can be:
- **Public** ("anyone with the URL can read"), or
- **Readable by your service account**. For Agent Engine deployments,
  grant the runtime SA `roles/storage.objectViewer` on the bucket:
  ```bash
  gcloud storage buckets add-iam-policy-binding gs://my-brand-pdfs \
      --member=serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com \
      --role=roles/storage.objectViewer
  ```

To pass via `deploy.py`:

```bash
python deploy/deploy.py \
    --project YOUR_PROJECT \
    --bucket gs://YOUR_STAGING_BUCKET \
    --brand-pdf-bucket my-brand-pdfs \
    --brand-pdf-prefix brands/
```

### Option 2 — Chat attachment (fallback)

Leave `BRAND_PDF_BUCKET` unset and the user attaches the PDF to their
chat message. The `find_brand_pdf` tool scans the session artifacts
and picks the most recently uploaded PDF.

Both paths feed the same downstream pipeline — `extract_brand_guidelines`
parses the chosen PDF into a structured brand profile regardless of
source.

Each standalone agent (`vto_agent/`, `marketing_campaign_agent/`) can also be
run on its own; see its package README.

### Verify the import path before launching

If you've moved things around or the checkout has an unexpected layout, run
this first — it loads the root agent the same way `adk` will:

```bash
python -c "import image_studio_root_agent.agent; print('OK')"
```

If it prints `OK`, `adk web image_studio_root_agent` will succeed.

## Region requirements

`GOOGLE_CLOUD_LOCATION` must be a region that supports **all** of:
`virtual-try-on-001`, `gemini-2.5-flash-image`, `imagen-4.0-generate-001`,
and `gemini-2.5-flash`. `us-central1` works for all four. Do **not** use
`global` — VTO requires a regional endpoint.

## Per-package docs

- [`vto_agent/README.md`](vto_agent/README.md) — VTO + Nano Banana details.
- [`marketing_campaign_agent/README.md`](marketing_campaign_agent/README.md) — campaign pipeline details and Google Ads spec mapping.
- [`image_studio_root_agent/README.md`](image_studio_root_agent/README.md) — how the tree is wired.
- [`docs/architecture.md`](docs/architecture.md) — design notes, state flow, and Google Ads spec reference.

## Deploy details

### Deploying the engine — `deploy/deploy.py`

| Mode | Command |
|---|---|
| **Create** (first deploy) | `python deploy/deploy.py --project P --bucket gs://B` |
| **Update** (subsequent) | `python deploy/deploy.py --project P --bucket gs://B --resource-id <ID>` |
| **Delete** | `python deploy/deploy.py --project P --delete --resource-id <ID>` |
| **Dry run** (no API call) | append `--dry-run` |
| **Smoke test after deploy** | append `--smoke-test` |

The script bundles the three sibling packages (`vto_agent/`,
`marketing_campaign_agent/`, `image_studio_root_agent/`) via
`extra_packages` pointing at their **outer** package directories — not the
inner module directories. A pre-flight check verifies each path contains
`<package>/agent.py` and aborts the deploy if any are missing, so you
won't ship a half-broken archive.

`requirements` is authoritative: Agent Engine installs only what's in
that list, ignoring your local `requirements.txt`. Add new runtime deps
to the `REQUIREMENTS` constant in `deploy.py`.

Env vars baked into the container: `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI=True`,
`CAMPAIGN_OUTPUT_DIR=/tmp/campaign_outputs`. Secrets should be wired via
Secret Manager — not in `env_vars`.

> Agent Engine doesn't auto-version. Subsequent deploys need the existing
> engine ID (printed on first deploy and shown in the console) or you'll
> spawn duplicates.

### Calling the deployed engine — `deploy/invoke_remote.py`

Three modes:

**Smoke test** — send a text message, watch streaming events:
```bash
python deploy/invoke_remote.py --resource-id <ID> \
    --message "hello, are you deployed?"
```

**Multi-turn** — reuse a session across calls (artifact + state persist):
```bash
python deploy/invoke_remote.py --resource-id <ID> \
    --session-id <session_id_from_a_previous_run> \
    --message "now build the campaign"
```

**End-to-end with file uploads** — attach person photo + product photo +
brand PDF and run the full pipeline:
```bash
python deploy/invoke_remote.py --resource-id <ID> \
    --attach person.jpg --attach shirt.jpg --attach brand.pdf \
    --message "try this shirt on the person, then build the campaign"
```

The script extracts any image data URIs from agent replies and writes
them to `--output-dir` (default `./remote_outputs/`), so you get the
generated images on disk after each run. Pass `--raw` for full event
dicts (base64 is elided to keep stdout readable).

### Building a frontend on top

For a custom UI (web app, Slack bot, Cloud Run service), use the same
SDK calls as `invoke_remote.py`:

```python
from vertexai import agent_engines
engine = agent_engines.get("projects/.../reasoningEngines/<ID>")
async for event in engine.async_stream_query(
    user_id="end-user-123",
    session_id=session_id,
    message=parts,  # list of {"inline_data": ...} + {"text": ...}
):
    ...
```

The image data URIs in the agent's text replies render directly in any
markdown viewer — no separate artifact fetching needed for the chat UI.

## Troubleshooting

### `ModuleNotFoundError: No module named 'vto_agent'`

`image_studio_root_agent/agent.py` imports from sibling packages. The
`sys.path` bootstrap at the top of that file walks up two parents to find
the repo root. If your checkout has an extra wrapping directory (e.g.
`image_studio_platform/image_studio_platform/...`), the bootstrap still
works because `parents[2]` is relative to the file itself, not the CWD.

Quick check — should print `True`:

```bash
python -c "
from pathlib import Path
import image_studio_root_agent
p = Path(image_studio_root_agent.__file__).resolve().parents[2]
print('repo root :', p)
print('siblings  :', (p / 'vto_agent' / 'vto_agent').exists(),
                     (p / 'marketing_campaign_agent' / 'marketing_campaign_agent').exists())
"
```

If `False`, your sibling packages aren't where the bootstrap expects.
Either restructure the checkout so `vto_agent/`, `marketing_campaign_agent/`,
and `image_studio_root_agent/` are direct children of the same parent, or
fall back to setting `PYTHONPATH` explicitly:

```bash
export PYTHONPATH="$PWD/vto_agent:$PWD/marketing_campaign_agent:$PWD/image_studio_root_agent:$PYTHONPATH"
```

### `Session not found ... The runner is configured with app name "X", but the root agent was loaded from "Y"`

The ADK CLI derives the session-store key from the package **directory name**,
not from `App(name=...)`. If they disagree, sessions are written under one
name and looked up under another.

Every `App(name=...)` in this repo already matches its directory:

| Directory | `App.name` |
|---|---|
| `vto_agent/vto_agent/` | `"vto_agent"` |
| `image_studio_root_agent/image_studio_root_agent/` | `"image_studio_root_agent"` |

`marketing_campaign_agent` uses `SequentialAgent` directly with no `App`
wrapper, so it inherits the directory name automatically. If you rename a
package directory, update its `App(name=...)` to match.

### `Publisher Model 'gemini-X-image-preview' was not found` (or similar 404)

Your GCP project doesn't have allowlist access to a preview image model.
Either request access on the Google AI Developers Forum, or fall back to
the GA model by setting in `.env`:

```
NANO_BANANA_MODEL=gemini-2.5-flash-image
```

This is the default and works on every Vertex-enabled project in a
supported region.

### `adk web` starts but the agent doesn't appear in the dropdown

Run `adk web` from the **directory above** the package, not from inside it.
From the repo root:

```bash
adk web image_studio_root_agent       # ✓
cd image_studio_root_agent && adk web # ✗ — won't find the package
```

## Known notes

- The agent doesn't render headline copy into images. Google Ads overlays
  text at serve time, and heavy embedded text reduces ad strength.
- VTO and Imagen outputs are SynthID-watermarked.
- For production, swap the in-memory artifact service for `GcsArtifactService`
  so generated images survive engine restarts.
- Imagen 4 doesn't natively render 4:1. The logo-landscape asset is
  generated at a wider native ratio and center-cropped to spec via Pillow
  (hence Pillow in the runtime requirements).
