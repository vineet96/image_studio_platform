"""ADK entry point for the virtual try-on / image-studio agent.

We intentionally do NOT import the agent module here. `adk web` /
`adk run` look up `vto_agent.agent.root_agent` directly, so this
package init can stay empty. Eagerly importing `agent` causes a
circular import when another package (e.g. image_studio_root_agent)
imports tool functions from `vto_agent.agent` while ADK is also
discovering this package as a top-level agent.
"""
