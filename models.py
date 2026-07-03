"""Pydantic models for structured compliance responses."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Deterministic confidence derived from official source count."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OfficialSource(BaseModel):
    """A citation from an official Swedish government website."""

    title: str | None = None
    url: str


class AgentAnswer(BaseModel):
    """Draft response produced by the LLM via output_schema."""

    summary: str
    answer: str
    official_sources: list[OfficialSource] = Field(default_factory=list)
    limitations: str = ""


class ComplianceResponse(AgentAnswer):
    """Final validated response returned to the user."""

    confidence: ConfidenceLevel
