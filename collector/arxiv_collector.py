"""Fetches and indexes arXiv papers authored by or citing tracked AI companies."""

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
import db
from entity_filter import check_entity

load_dotenv()

SEED_FILE = Path(__file__).parent / "seed_companies.json"
RATE_LIMIT_SLEEP = 3  # seconds between companies; ArXiv is strict
LOOKBACK_DAYS = 90

# arXiv indexes stemmed terms, so `all:"Cohere"` also matches "coherence" and
# "coherent" — which pulled quantum-physics papers into an AI vendor's
# technology momentum. Papers must therefore sit in an AI/CS-relevant primary
# category AND name the company as a whole word (see check_entity).
ALLOWED_CATEGORY_PREFIXES = ("cs.",)
ALLOWED_CATEGORIES = frozenset({"stat.ML", "eess.AS", "eess.IV"})


# ---------------------------------------------------------------------------
# ArXiv API interaction
# ---------------------------------------------------------------------------

def search_arxiv(company_name: str, max_results: int = 20) -> str:
    """Query the arXiv API and return raw XML response.

    Searches title and abstract only. The previous `all:"{name}"` searched full
    text including the bibliography, so it returned papers that merely *cite* a
    vendor — or that use its name as an ordinary word. arXiv's stemmed index
    made that far worse: "Anthropic" matched cosmology papers about the
    anthropic principle, "Modal" matched anything multimodal, and "Cohere"
    matched every paper mentioning coherence. 85% of collected papers were
    unrelated to the vendor they were filed under.
    """
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f'ti:"{company_name}" OR abs:"{company_name}"',
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def parse_arxiv_response(xml_text: str) -> list[dict]:
    """Parse arXiv XML response and extract paper metadata."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall("atom:entry", ns):
        title_elem = entry.find("atom:title", ns)
        summary_elem = entry.find("atom:summary", ns)
        published_elem = entry.find("atom:published", ns)
        id_elem = entry.find("atom:id", ns)
        author_elems = entry.findall("atom:author", ns)

        title = title_elem.text.strip() if title_elem is not None else ""
        summary = summary_elem.text.strip() if summary_elem is not None else ""
        published = published_elem.text.strip() if published_elem is not None else ""
        entry_id = id_elem.text.strip() if id_elem is not None else ""

        authors = []
        affiliations = []
        for author in author_elems:
            name_elem = author.find("atom:name", ns)
            if name_elem is not None:
                authors.append(name_elem.text)
            # Affiliation is optional in the arXiv schema, but when present it
            # is the strongest evidence that a paper really is the company's.
            for affil in author.findall("arxiv:affiliation", ns):
                if affil.text:
                    affiliations.append(affil.text.strip())

        primary_elem = entry.find("arxiv:primary_category", ns)
        primary_category = (
            primary_elem.get("term") if primary_elem is not None else None
        )
        categories = [
            cat.get("term")
            for cat in entry.findall("atom:category", ns)
            if cat.get("term")
        ]

        papers.append({
            "title": title,
            "summary": summary,
            "published": published,
            "entry_id": entry_id,
            "authors": authors[:5],
            "affiliations": affiliations[:5],
            "primary_category": primary_category,
            "categories": categories,
        })

    return papers


# ---------------------------------------------------------------------------
# Paper filtering
# ---------------------------------------------------------------------------

def _is_recent(published_str: str) -> bool:
    """Check if a paper was published within the lookback window."""
    if not published_str:
        return False
    try:
        pub_date = datetime.fromisoformat(
            published_str.replace("Z", "+00:00")
        )
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)
        return pub_date >= cutoff
    except ValueError:
        return False


def _parse_published_date(published_str: str):
    """Parse arXiv ISO date string to a Python date object."""
    return datetime.fromisoformat(
        published_str.replace("Z", "+00:00")
    ).date()


def _is_ai_relevant(paper: dict) -> bool:
    """Check that a paper sits in an AI/CS-relevant arXiv category.

    Uses the primary category when arXiv supplies one, falling back to the full
    category list. This is what keeps astro-ph, math and quant-ph papers — which
    the stemmed search matches on "coherence" — out of technology momentum.
    """
    primary = paper.get("primary_category")
    candidates = [primary] if primary else paper.get("categories", [])
    return any(
        cat in ALLOWED_CATEGORIES or cat.startswith(ALLOWED_CATEGORY_PREFIXES)
        for cat in candidates
        if cat
    )


# ---------------------------------------------------------------------------
# Per-company processing
# ---------------------------------------------------------------------------

def _process_company(company: dict) -> tuple[int, int, int]:
    """Search arXiv for papers mentioning a company.

    Returns (signals_added, errors, rejected) where `rejected` counts papers
    dropped as off-topic or belonging to a different entity.
    """
    name = company["name"]
    ticker = company.get("ticker")
    signals_added = 0
    errors = 0
    rejected = 0

    try:
        xml_response = search_arxiv(name, max_results=20)
        papers = parse_arxiv_response(xml_response)
    except Exception as exc:
        print(f"  ArXiv search failed for {name}: {exc}")
        return 0, 1, 0

    for paper in papers:
        if not _is_recent(paper["published"]):
            continue

        if not _is_ai_relevant(paper):
            rejected += 1
            print(
                f"    [skip: category {paper.get('primary_category')}] "
                f"{paper['title'][:70]}"
            )
            continue

        # Confirm the paper actually names this company (or an affiliation does)
        # rather than merely stemming to it.
        verdict = check_entity(
            company, paper["title"], paper["summary"], *paper["affiliations"]
        )
        if not verdict.ok:
            rejected += 1
            print(f"    [skip: {verdict.reason}] {paper['title'][:70]}")
            continue

        headline = paper["title"][:200]
        summary = paper["summary"][:300]
        pub_date = _parse_published_date(paper["published"])

        try:
            inserted = db.insert_signal(
                company_name=name,
                ticker=ticker,
                signal_type="research_paper",
                signal_date=pub_date,
                headline=headline,
                summary=summary,
                source_url=paper["entry_id"],
                importance_score=60,
                raw_data={
                    "arxiv_id": paper["entry_id"],
                    "authors": paper["authors"],
                    "affiliations": paper["affiliations"],
                    "primary_category": paper["primary_category"],
                    "categories": paper["categories"],
                },
            )
            if inserted:
                signals_added += 1
        except Exception as exc:
            print(f"    [insert error] {headline}: {exc}")
            errors += 1

    return signals_added, errors, rejected

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    with open(SEED_FILE) as f:
        all_companies = json.load(f)

    # ArXiv search works on company names, not specific to public/private
    companies = all_companies
    total = len(companies)
    total_signals = 0
    total_errors = 0
    total_rejected = 0
    started_at = datetime.now(tz=timezone.utc)

    print(f"ArXiv collector starting — {total} companies to process.\n")

    for i, company in enumerate(companies, start=1):
        name = company["name"]
        print(f"Processing {name} ({i}/{total})...")

        try:
            signals, errors, rejected = _process_company(company)
            total_signals += signals
            total_errors += errors
            total_rejected += rejected
            print(
                f"  -> {signals} signal(s) added, {rejected} rejected "
                f"(off-topic/wrong entity), {errors} error(s)."
            )
        except Exception as exc:
            print(f"  ERROR processing {name}: {exc}")
            total_errors += 1

        if i < total:
            time.sleep(RATE_LIMIT_SLEEP)

    db.log_run(
        collector_name="arxiv_collector",
        companies_processed=total,
        signals_added=total_signals,
        errors=total_errors,
        started_at=started_at,
    )

    print(
        f"\nDone. {total_signals} signals added, {total_rejected} rejected as "
        f"off-topic or wrong-entity, {total_errors} errors."
    )


if __name__ == "__main__":
    run()
