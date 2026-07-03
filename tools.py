"""Tools and domain constants for the compliance assistant."""

from __future__ import annotations

from google.adk.tools import google_search

OFFICIAL_DOMAINS: tuple[str, ...] = (
    "skatteverket.se",
    "bolagsverket.se",
    "verksamt.se",
)

# Common third-party titles returned by Google Search grounding (not government).
NON_OFFICIAL_SOURCE_HINTS: tuple[str, ...] = (
    "avalara",
    "taxually",
    "eurotax",
    "wise.com",
    "pwc.com",
    "deloitte",
    "kpmg",
    "lawline",
    "lexnova",
    "viestinn",
    "wolters",
    "accigo",
    "hogia",
)

SEARCH_QUERY_TEMPLATE = (
    "(site:skatteverket.se OR site:bolagsverket.se OR site:verksamt.se) {query}"
)

TOOLS = [google_search]


def search_keywords_for_query(query: str) -> str:
    """Prefer Swedish keywords for official-site search when possible."""
    lower = query.lower()
    if any(
        term in lower
        for term in (
            "hembud",
            "aktieägaravtal",
            "shareholder agreement",
            "bolagsordning",
            "aktiebolag",
            "förköps",
        )
    ):
        return "aktieägaravtal hembudsförbehåll bolagsordning AB"
    if any(term in lower for term in ("vat", "moms", "tax", "skatt")):
        return "moms mervärdesskatt"
    if any(term in lower for term in ("register", "registr", "starta bolag")):
        return "registrera aktiebolag"
    return query.strip()


def build_official_search_query(query: str) -> str:
    """Format a user question as a domain-restricted search query."""
    return SEARCH_QUERY_TEMPLATE.format(query=search_keywords_for_query(query))
