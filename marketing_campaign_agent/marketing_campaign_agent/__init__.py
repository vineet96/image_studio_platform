"""Marketing campaign agent package.

Empty by design — see note in vto_agent/__init__.py. `adk` discovers
`marketing_campaign_agent.agent.root_agent` directly, and the root
agent package imports `marketing_campaign_agent` from
`marketing_campaign_agent.agent` (explicit subpath, not from the
package init). Keeping this empty avoids circular imports during
ADK's multi-package discovery.
"""
