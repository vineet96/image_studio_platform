"""Image Studio root agent: VTO + edit_image tools plus marketing
campaign sub-agent.

Empty by design — see note in vto_agent/__init__.py. ADK CLI looks up
`image_studio_root_agent.agent.root_agent` and
`image_studio_root_agent.agent.app` directly, so eager imports here
would only add risk of circular-import collisions during multi-package
discovery.

For deployment (deploy/deploy.py), import the root_agent explicitly
from `image_studio_root_agent.agent`.
"""
