"""Entity disambiguation for collected signals.

A seed company name is not a unique identifier. "Cohere" the AI/LLM vendor
shares its name with Cohere Health (health insurance) and Cohere Technologies
(wireless RF); arXiv's stemmed index matches "Cohere" against "coherence"; and
names like Runway or Harvey are ordinary English words. Without a check, a
signal about a different organisation gets stored under the tracked vendor and
silently becomes evidence in its brief — a $28M Department of War contract
awarded to *Cohere Technologies* lifted the AI vendor's Financial Health score
by 25 points, and four quantum-physics papers matched on the word "coherence"
were counted as its technology momentum.

Every collector routes candidate signals through `check_entity` before insert.
The check is deterministic and configured per company in seed_companies.json:

  aliases          - accepted surface forms of the name. At least one must
                     appear as a whole word in the text. Defaults to [name].
  exclude_entities - names of colliding organisations. Any match rejects.
  exclude_terms    - subject-matter terms belonging to a colliding entity's
                     domain rather than this vendor's. Any match rejects.

Precision is preferred over recall: in a procurement brief a dropped signal is
a visible gap (confidence scoring already flags thin evidence), but a
wrong-entity signal is a false claim presented as sourced fact.

Known limitation: common-word names (Adept, Writer, Notion, Glean, Modal,
Replicate) can still admit unrelated text, because requiring a longer alias
would drop legitimate coverage. The LLM `about_company` check in
news_collector is the second line of defence for those.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

SEED_FILE = Path(__file__).parent / "seed_companies.json"


class EntityCheck(NamedTuple):
    """Result of verifying that a text is about the expected company."""

    ok: bool
    reason: str


@lru_cache(maxsize=2048)
def _pattern(phrase: str) -> re.Pattern:
    """Compile a whole-word, case-insensitive matcher for a phrase.

    Internal whitespace matches any run of whitespace, so "Cohere  Health" and
    "Cohere\\nHealth" both match "Cohere Health". The lookarounds are `\\w`
    based rather than `\\b` so that phrases ending in punctuation (e.g.
    "Otter.ai", "W&B") still anchor correctly.
    """
    escaped = re.escape(phrase.strip())
    # re.escape escapes spaces on Python < 3.7 but not on 3.7+; handle both.
    escaped = re.sub(r"(?:\\\s|\s)+", r"\\s+", escaped)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _matches(phrase: str, haystack: str) -> bool:
    return bool(_pattern(phrase).search(haystack))


def aliases_for(company: dict) -> list[str]:
    """Accepted surface forms of a company's name, defaulting to the name."""
    aliases = company.get("aliases") or []
    return aliases if aliases else [company["name"]]


@lru_cache(maxsize=1)
def load_company_index(seed_file: str = str(SEED_FILE)) -> dict[str, dict]:
    """Map company name -> seed config, for collectors that only have a name."""
    with open(seed_file) as f:
        companies = json.load(f)
    return {c["name"]: c for c in companies}


def check_entity(company: dict, *texts: str | None) -> EntityCheck:
    """Verify that the given text is about `company` and not a namesake.

    Pass every text you have (headline, summary, author affiliations). Checks
    run in order: colliding organisation names, then off-domain terms, then
    whether the company is actually named at all. Order matters — "Cohere
    Health" contains "Cohere", so the exclusion must be tested first.
    """
    haystack = " ".join(t for t in texts if t)
    if not haystack.strip():
        return EntityCheck(False, "no text to verify against")

    for phrase in company.get("exclude_entities", []):
        if _matches(phrase, haystack):
            return EntityCheck(
                False, f'mentions "{phrase}", a different organisation'
            )

    for term in company.get("exclude_terms", []):
        if _matches(term, haystack):
            return EntityCheck(
                False, f'off-domain term "{term}" indicates a different entity'
            )

    names = aliases_for(company)
    if not any(_matches(alias, haystack) for alias in names):
        return EntityCheck(
            False, f'no whole-word mention of "{company["name"]}" or its aliases'
        )

    return EntityCheck(True, "ok")


def news_query(company: dict) -> str:
    """Build a Google News search query that excludes known namesakes.

    Google News RSS honours `-"phrase"` exclusions, so filtering upstream costs
    nothing and keeps colliding organisations out of the feed entirely.
    """
    name = company["name"]
    query = f'"{name}" AI'
    for phrase in company.get("exclude_entities", []):
        query += f' -"{phrase}"'
    return query
