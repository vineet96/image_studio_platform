# Image Studio ADK Agent (VTO + Nano Banana)

Standalone agent with two tools:

| Capability | Model | Tool |
|---|---|---|
| Virtual try-on | `virtual-try-on-001` | `virtual_try_on` |
| Generative image editing | Nano Banana (`gemini-2.5-flash-image`) | `edit_image` |

This package is one of three in the [Image Studio Platform](../README.md).
For the root agent that also produces Google Ads campaigns, see
[`../image_studio_root_agent/`](../image_studio_root_agent/).

## Layout

```
vto_agent/
├── requirements.txt
└── vto_agent/
    ├── __init__.py
    ├── agent.py            # root_agent + app + 2 tools
    └── .env.example
```

## Run

```bash
pip install -r requirements.txt
gcloud auth application-default login
cp vto_agent/.env.example vto_agent/.env   # fill in GOOGLE_CLOUD_PROJECT
adk web                                     # from this directory
```
