# Image Studio Root Agent

The end-to-end agent: VTO + Nano Banana editing **plus** the marketing
campaign sub-agent. This is the package you deploy to production.

## What it does

```
User                                    Image Studio root agent (gemini-2.5-flash)
─────                                   ──────────────────────────────────────
upload person + product, "try this" →   virtual_try_on tool  →  VTO image inline
"change the shirt to navy"          →   edit_image tool      →  edited image inline
upload brand.pdf, "make Google ads" →   transfer to marketing_campaign_agent:
                                          brand_parser  → state['brand']
                                          creative_director → state['creative_plan']
                                          asset_generator → 4 Google RDA assets inline
```

## Layout

```
image_studio_root_agent/
├── requirements.txt
└── image_studio_root_agent/
    ├── __init__.py
    ├── agent.py            # root_agent (tools + sub_agents) + app
    └── .env.example
```

`image_studio_root_agent/agent.py` imports `virtual_try_on` and `edit_image` from
the sibling `vto_agent` package, and `marketing_campaign_agent` from the
sibling campaign package. All three must be on `PYTHONPATH`.

## Run locally

From the repo root:

```bash
pip install -r image_studio_root_agent/requirements.txt

# Make the sibling packages importable
export PYTHONPATH="$PWD/vto_agent:$PWD/marketing_campaign_agent:$PWD/image_studio_root_agent"

cp image_studio_root_agent/image_studio_root_agent/.env.example image_studio_root_agent/image_studio_root_agent/.env
# edit .env: set GOOGLE_CLOUD_PROJECT

adk web image_studio_root_agent     # or: adk web   (from inside image_studio_root_agent/)
```

## Deploy

See [`../deploy/`](../deploy/).
