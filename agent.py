"""Root agent definition for the Swedish Compliance AI Assistant."""

from __future__ import annotations

from google.adk.agents import Agent

from .config import validate_api_key
from .prompts import SYSTEM_INSTRUCTION
from .utils import before_agent_callback, setup_logging

setup_logging()
validate_api_key()

# The grounded search + structured response is produced entirely in
# ``before_agent_callback`` with a single site-restricted search call, which
# ends the invocation before the model runs. The instruction below documents
# the assistant's contract and acts as a safety net for non-question turns.
root_agent = Agent(
    name="swedish_compliance_assistant",
    model="gemini-2.5-flash-lite",
    description=(
        "Swedish SME compliance assistant grounded on official government sources."
    ),
    instruction=SYSTEM_INSTRUCTION,
    before_agent_callback=before_agent_callback,
)
