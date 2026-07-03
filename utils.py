"""Grounding validation, confidence scoring, logging, and agent callbacks."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from .models import ComplianceResponse, ConfidenceLevel, OfficialSource
from .tools import NON_OFFICIAL_SOURCE_HINTS, OFFICIAL_DOMAINS, build_official_search_query

_OFFICIAL_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:skatteverket|bolagsverket|verksamt)\.se[^\s\)\]\"'<>]*",
    re.IGNORECASE,
)
_OFFICIAL_MARKDOWN_URL_PATTERN = re.compile(
    r"\[[^\]]*\]\((https?://(?:www\.)?(?:skatteverket|bolagsverket|verksamt)\.se[^)]*)\)",
    re.IGNORECASE,
)

logger = logging.getLogger("swedish_compliance")

FALLBACK_MESSAGE = (
    "I could not find sufficient information from the official Swedish "
    "authorities to answer this confidently."
)

# Tried in order; on 503 (overload) or 429 (quota) we fall back to the next.
# Each model has its own free-tier daily quota, so this also spreads load.
SEARCH_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of post-generation grounding checks."""

    is_valid: bool
    has_official_sources: bool
    has_citations: bool
    reason: str = ""


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def is_official_url(url: str) -> bool:
    """Return True if the URL belongs to an allowed government domain."""
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return any(host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_DOMAINS)
    except Exception:
        return False


def filter_official_sources(sources: list[OfficialSource]) -> list[OfficialSource]:
    """Keep only sources from official Swedish government domains."""
    filtered: list[OfficialSource] = []
    for source in sources:
        url = str(source.url)
        title = source.title or ""
        combined = f"{title} {url}"
        if is_non_official_source_hint(combined):
            continue
        if is_official_url(url):
            filtered.append(source)
            continue
        if official_domain_from_text(combined):
            filtered.append(
                OfficialSource(
                    title=title or official_domain_from_text(combined),
                    url=resolve_official_source_url(title, url) or url,
                )
            )
    return filtered


def is_non_official_source_hint(text: str) -> bool:
    """Return True when a title or URL clearly belongs to a third-party site."""
    lowered = text.lower()
    return any(hint in lowered for hint in NON_OFFICIAL_SOURCE_HINTS)


def official_domain_from_text(text: str) -> str | None:
    """Return the matching official domain name found in free text."""
    lowered = text.lower()
    for domain in OFFICIAL_DOMAINS:
        stem = domain.split(".")[0]
        if domain in lowered or stem in lowered:
            return domain
    return None


def canonical_official_url(domain: str) -> str:
    """Return a stable homepage URL for an official domain."""
    return f"https://www.{domain}/"


def resolve_official_source_url(title: str | None, uri: str | None) -> str | None:
    """Pick the best citation URL for an official grounding chunk."""
    if uri and is_official_url(uri):
        return uri.strip()

    domain = official_domain_from_text(f"{title or ''} {uri or ''}")
    if domain:
        return canonical_official_url(domain)

    return uri.strip() if uri else None


def is_official_grounding_chunk(chunk: types.GroundingChunk) -> bool:
    """Return True when a grounding chunk likely comes from an official source."""
    if not chunk.web or not chunk.web.uri:
        return False

    title = chunk.web.title or ""
    uri = chunk.web.uri or ""
    combined = f"{title} {uri}".lower()

    if is_non_official_source_hint(combined):
        return False

    if is_official_url(uri):
        return True

    domain = (chunk.web.domain or "").lower()
    if domain and any(
        domain == official or domain.endswith(f".{official}")
        for official in OFFICIAL_DOMAINS
    ):
        return True

    return official_domain_from_text(title) is not None


def extract_sources_from_grounding(
    metadata: types.GroundingMetadata | None,
    *,
    trust_site_restricted: bool = False,
) -> list[OfficialSource]:
    """Extract official citation URLs from Gemini grounding metadata."""
    if not metadata or not metadata.grounding_chunks:
        return []

    sources: list[OfficialSource] = []
    seen: set[str] = set()

    for chunk in metadata.grounding_chunks:
        if not chunk.web or not chunk.web.uri:
            continue

        title = chunk.web.title or ""
        uri = chunk.web.uri or ""
        combined = f"{title} {uri}"

        is_official = is_official_grounding_chunk(chunk)
        if not is_official and trust_site_restricted:
            is_official = not is_non_official_source_hint(combined)

        if not is_official:
            continue

        url = resolve_official_source_url(title, uri)
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(OfficialSource(title=title or official_domain_from_text(combined), url=url))

    return sources


def extract_official_urls_from_text(text: str) -> list[OfficialSource]:
    """Extract explicit official URLs cited in text (plain or markdown links)."""
    sources: list[OfficialSource] = []
    seen: set[str] = set()

    for pattern in (
        _OFFICIAL_MARKDOWN_URL_PATTERN,
        _OFFICIAL_URL_PATTERN,
    ):
        for match in pattern.finditer(text):
            url = match.group(1) if match.lastindex else match.group(0)
            url = url.rstrip(".,;)")
            if url in seen:
                continue
            seen.add(url)
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            sources.append(OfficialSource(title=domain, url=url))

    return sources


def merge_sources(
    llm_sources: list[OfficialSource],
    grounding_sources: list[OfficialSource],
) -> list[OfficialSource]:
    """Merge LLM citations with grounding chunks; prefer specific page URLs."""
    merged: dict[str, OfficialSource] = {}
    for source in grounding_sources + llm_sources:
        url = str(source.url).strip()
        if not url:
            continue
        if url not in merged or (source.title and not merged[url].title):
            merged[url] = source

    # Drop bare homepages when a deeper page exists for the same domain.
    by_domain: dict[str, list[OfficialSource]] = {}
    for source in merged.values():
        domain = urlparse(str(source.url)).netloc.lower().removeprefix("www.")
        by_domain.setdefault(domain, []).append(source)

    result: list[OfficialSource] = []
    for domain, domain_sources in by_domain.items():
        if len(domain_sources) == 1:
            result.append(domain_sources[0])
            continue
        specific = [
            source
            for source in domain_sources
            if urlparse(str(source.url)).path not in ("", "/")
        ]
        result.extend(specific or domain_sources[:1])

    return result


def _is_specific_page(url: str) -> bool:
    """Return True for a deep official page (not a bare homepage)."""
    return urlparse(url).path.strip("/") != ""


def compute_confidence(sources: list[OfficialSource]) -> ConfidenceLevel:
    """Derive confidence from distinct *specific* official pages.

    Bare homepages (e.g. https://www.skatteverket.se/) do not count toward
    MEDIUM/HIGH because they are not real citations for a specific claim.
    """
    official = filter_official_sources(sources)
    specific_urls = {
        str(source.url) for source in official if _is_specific_page(str(source.url))
    }

    if len(specific_urls) >= 2:
        return ConfidenceLevel.HIGH
    if len(specific_urls) == 1:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def is_refusal_answer(text: str) -> bool:
    """Return True when the model answer is (essentially) the refusal message."""
    if not text or not text.strip():
        return True
    normalized = text.strip().lower()
    return normalized.startswith(
        "i could not find sufficient information from the official swedish"
    )


def validate_grounding(
    official_sources: list[OfficialSource],
    answer_text: str,
) -> ValidationResult:
    """Verify that the response is backed by official government sources."""
    has_official = len(official_sources) > 0
    has_citations = has_official
    has_answer = bool(answer_text and answer_text.strip())

    if not has_official:
        return ValidationResult(
            is_valid=False,
            has_official_sources=False,
            has_citations=False,
            reason="No official government sources retrieved.",
        )

    if not has_answer:
        return ValidationResult(
            is_valid=False,
            has_official_sources=True,
            has_citations=True,
            reason="Answer text is empty despite retrieved sources.",
        )

    return ValidationResult(
        is_valid=True,
        has_official_sources=True,
        has_citations=True,
        reason="Response grounded on official sources.",
    )


def build_fallback_response(limitations: str | None = None) -> ComplianceResponse:
    """Safe response when grounding validation fails."""
    return ComplianceResponse(
        summary=FALLBACK_MESSAGE,
        answer=FALLBACK_MESSAGE,
        official_sources=[],
        limitations=limitations
        or "No official sources from skatteverket.se, bolagsverket.se, or verksamt.se were found.",
        confidence=ConfidenceLevel.LOW,
    )


def format_compliance_response(response: ComplianceResponse) -> str:
    """Serialize the final response as pretty-printed JSON."""
    return json.dumps(response.model_dump(), indent=2, ensure_ascii=False)


def build_summary(answer: str, max_len: int = 220) -> str:
    """Build a short summary from the full agent answer."""
    text = answer.strip()
    if not text:
        return ""

    for separator in (". ", ".\n", "\n\n"):
        if separator in text:
            candidate = text.split(separator, 1)[0] + separator.strip()
            if len(candidate) <= max_len:
                return candidate

    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def build_limitations(confidence: ConfidenceLevel, validation: ValidationResult) -> str:
    """Describe caveats for the final structured response."""
    if confidence == ConfidenceLevel.HIGH:
        return (
            "Based on multiple official Swedish government sources. "
            "Verify critical decisions with the relevant authority or a professional."
        )
    if confidence == ConfidenceLevel.MEDIUM:
        return (
            "Based on a single official source. Cross-check with Skatteverket, "
            "Bolagsverket, or Verksamt before acting."
        )
    return validation.reason or (
        "No official sources from skatteverket.se, bolagsverket.se, or verksamt.se were found."
    )


def extract_last_user_message(events: list[Any]) -> str:
    """Return the most recent user message from session events."""
    for event in reversed(events):
        if event.author != "user" or not event.content or not event.content.parts:
            continue
        texts = [part.text for part in event.content.parts if part.text]
        if texts:
            return "".join(texts)
    return ""


_SEARCH_ASSISTANT_INSTRUCTION = (
    "You are a search assistant for Swedish authorities. Call the google_search "
    "tool with the user's message as the query. Then respond in this exact format:\n\n"
    "SOURCES:\n"
    "<one full https:// URL per line, ONLY from skatteverket.se, bolagsverket.se, "
    "or verksamt.se; deep pages preferred over homepages>\n\n"
    "SUMMARY:\n"
    "<concise summary of the relevant facts found on those official pages>\n\n"
    "If you cannot find any relevant page on those three official domains, reply "
    "with exactly: NO_OFFICIAL_SOURCES"
)


_OVERLOAD_HINTS = ("503", "unavailable", "deadline", "timeout", "500", "internal")
_QUOTA_HINTS = ("429", "quota", "resource_exhausted")


def _is_overloaded(exc: Exception) -> bool:
    """Return True for transient server overload/network errors."""
    text = str(exc).lower()
    return any(hint in text for hint in _OVERLOAD_HINTS)


def _is_quota_error(exc: Exception) -> bool:
    """Return True when a model's quota is exhausted (skip to next model)."""
    text = str(exc).lower()
    return any(hint in text for hint in _QUOTA_HINTS)


def _generate_one_model(client: Any, model: str, search_query: str, attempts: int) -> Any:
    """Call a single model, retrying overload errors with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=search_query,
                config=types.GenerateContentConfig(
                    system_instruction=_SEARCH_ASSISTANT_INSTRUCTION,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
        except Exception as exc:
            if attempt >= attempts or not _is_overloaded(exc):
                raise
            backoff = 2 ** (attempt - 1)
            logger.warning(
                "%s overloaded (attempt %d/%d): %s. Retrying in %ds.",
                model,
                attempt,
                attempts,
                exc,
                backoff,
            )
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def _generate_with_fallback(client: Any, search_query: str, attempts: int = 3) -> Any:
    """Try each model in ``SEARCH_MODELS`` until one succeeds.

    Overload (503) or quota (429) on one model triggers a switch to the next.
    The last error is re-raised if every model fails.
    """
    last_exc: Exception | None = None
    for model in SEARCH_MODELS:
        try:
            return _generate_one_model(client, model, search_query, attempts)
        except Exception as exc:
            last_exc = exc
            if _is_overloaded(exc) or _is_quota_error(exc):
                logger.warning("Model %s unavailable (%s); trying next model.", model, exc)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def run_official_search(query: str) -> tuple[list[OfficialSource], str]:
    """Run a site-restricted Google Search and return official sources plus context."""
    from google.genai import Client

    search_query = build_official_search_query(query)
    client = Client()

    response = _generate_with_fallback(client, search_query)

    metadata = None
    if response.candidates:
        metadata = response.candidates[0].grounding_metadata

    context_text = (response.text or "").strip()
    if context_text.strip().upper().startswith("NO_OFFICIAL_SOURCES"):
        logger.info("Search assistant reported NO_OFFICIAL_SOURCES.")
        return [], ""

    # Real citations come from explicit URLs in the summary text (grounding
    # chunks on the Gemini API are opaque redirect links).
    sources = extract_official_urls_from_text(context_text)
    sources = merge_sources(
        sources,
        extract_sources_from_grounding(metadata, trust_site_restricted=False),
    )
    sources = filter_official_sources(sources)

    logger.info("Programmatic search query: %s", search_query)
    logger.info("Programmatic search official sources: %d", len(sources))
    return sources, context_text


def parse_search_summary(context_text: str) -> str:
    """Extract the SUMMARY section from the search assistant output."""
    if not context_text:
        return ""
    upper = context_text.upper()
    marker = "SUMMARY:"
    idx = upper.find(marker)
    if idx >= 0:
        return context_text[idx + len(marker) :].strip()
    # No explicit marker: drop a leading SOURCES block if present.
    src_idx = upper.find("SOURCES:")
    if src_idx >= 0:
        remainder = context_text[src_idx + len("SOURCES:") :]
        lines = [line for line in remainder.splitlines() if "http" not in line.lower()]
        return "\n".join(lines).strip()
    return context_text.strip()


def build_compliance_response(
    sources: list[OfficialSource],
    context_text: str,
) -> ComplianceResponse:
    """Compose the final structured response from grounded search results."""
    official_sources = filter_official_sources(sources)
    answer = parse_search_summary(context_text)
    confidence = compute_confidence(official_sources)
    validation = validate_grounding(official_sources, answer)

    if (
        is_refusal_answer(answer)
        or not validation.is_valid
        or confidence == ConfidenceLevel.LOW
    ):
        logger.info("Falling back: reason=%s confidence=%s", validation.reason, confidence)
        return build_fallback_response(validation.reason)

    return ComplianceResponse(
        summary=build_summary(answer),
        answer=answer,
        official_sources=official_sources,
        limitations=build_limitations(confidence, validation),
        confidence=confidence,
    )


def describe_search_error(exc: Exception) -> str:
    """Turn a raised exception into an informative limitations message."""
    text = str(exc).lower()
    if "429" in text or "resource_exhausted" in text or "quota" in text:
        return (
            "Search is temporarily unavailable: the Gemini API quota/rate limit was "
            "reached. Please retry later or enable billing for a higher quota."
        )
    if "401" in text or "unauthenticated" in text or "permission" in text:
        return "Search failed: invalid or missing GOOGLE_API_KEY. Check your .env file."
    if "503" in text or "unavailable" in text or "timeout" in text or "deadline" in text:
        return "Search is temporarily unavailable (network/service error). Please retry."
    return "Search failed due to an unexpected error. Please retry."


async def before_agent_callback(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Search official sources, compose the grounded answer, and short-circuit.

    A single site-restricted LLM search call produces both the citations and the
    summary. Returning content here ends the invocation, so the root agent's
    model is never called again (one LLM call per question, no duplicate output).
    """
    query = extract_last_user_message(callback_context.session.events)
    if not query:
        return None

    logger.info("Incoming query: %s", query)
    start = time.perf_counter()
    try:
        sources, context_text = run_official_search(query)
        logger.info("Official sources retrieved: %d", len(sources))
        response = build_compliance_response(sources, context_text)
    except Exception as exc:
        logger.exception("Compliance pipeline failed; returning safe fallback.")
        response = build_fallback_response(describe_search_error(exc))

    logger.info("Generation time: %.0f ms", (time.perf_counter() - start) * 1000)
    logger.info("Final confidence: %s", response.confidence.value)
    return types.Content(
        role="model",
        parts=[types.Part(text=format_compliance_response(response))],
    )
