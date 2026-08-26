"""Unit tests for the brief evaluation checks and the UNKNOWN honesty guard.

These exercise the pure check logic against fixture briefs, so they need no
database, no AWS credentials and no network. That is what makes them safe to
run in CI, where the full eval (which generates real briefs) cannot run.

Run with:
    python -m unittest discover -s evaluation -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor import (
    UNKNOWN_LINE,
    _dimensions_without_evidence,
    _enforce_unknown_sections,
)
from evaluation.eval_runner import (
    _normalize_amount,
    check_amount_grounding,
    check_date_grounding,
    check_filler,
    check_structure,
    split_sections,
)

# A brief in the shape the synthesiser actually emits, markdown bold included.
FIXTURE_BRIEF = """**VENDOR INTELLIGENCE BRIEF: Acme AI**

**FINANCIAL HEALTH:** 65/100
Acme raised $150M on 2026-05-12.

**TECHNOLOGY MOMENTUM:** 85/100
75 GitHub releases in the last 90 days.

**NEWS & SENTIMENT:** Negative
Coverage on 2026-07-08 flagged security concerns.

**PERSONNEL STABILITY:** 70/100
Hired Teresa Carlson on 2026-07-08.

**COMPETITIVE POSITION:**
Acme is a prominent player in the AI development space.

**EXECUTIVE SUMMARY:**
Recommendation: Recommended with caution.

**DATA SOURCES:** SEC EDGAR, GitHub
"""

FULL_EVIDENCE = {
    "financial": {"level": "high", "evidence_count": 73},
    "technology": {"level": "high", "evidence_count": 19},
    "news": {"level": "high", "evidence_count": 168},
    "personnel": {"level": "high", "evidence_count": 4},
    "competitive": {"level": "high", "evidence_count": 3},
}


def _confidence(**overrides) -> dict:
    dimensions = {k: dict(v) for k, v in FULL_EVIDENCE.items()}
    dimensions.update(overrides)
    return {"dimensions": dimensions}


class TestSectionParsing(unittest.TestCase):
    def test_finds_every_section(self):
        sections = split_sections(FIXTURE_BRIEF)
        for name in (
            "FINANCIAL HEALTH",
            "TECHNOLOGY MOMENTUM",
            "NEWS & SENTIMENT",
            "PERSONNEL STABILITY",
            "COMPETITIVE POSITION",
            "EXECUTIVE SUMMARY",
        ):
            self.assertIn(name, sections)

    def test_body_excludes_the_next_header(self):
        sections = split_sections(FIXTURE_BRIEF)
        self.assertIn("$150M", sections["FINANCIAL HEALTH"])
        self.assertNotIn("TECHNOLOGY MOMENTUM", sections["FINANCIAL HEALTH"])

    def test_structure_check_reports_missing_section(self):
        stripped = FIXTURE_BRIEF.replace("**PERSONNEL STABILITY:** 70/100", "")
        failures = check_structure(split_sections(stripped))
        self.assertTrue(any("PERSONNEL STABILITY" in f for f in failures))

    def test_structure_check_passes_complete_brief(self):
        self.assertEqual(check_structure(split_sections(FIXTURE_BRIEF)), [])


class TestFillerCheck(unittest.TestCase):
    def test_flags_banned_phrase(self):
        failures = check_filler(split_sections(FIXTURE_BRIEF))
        self.assertTrue(any("prominent player" in f for f in failures))

    def test_passes_specific_prose(self):
        sections = {"COMPETITIVE POSITION": "Competes with Cohere and Mistral."}
        self.assertEqual(check_filler(sections), [])


class TestGroundingChecks(unittest.TestCase):
    """A grounding check that cannot fail is worse than no check, so both
    directions are asserted."""

    dates = {"2026-05-12", "2026-07-08"}
    amounts = {150.0, 1200.0}

    def test_flags_date_absent_from_evidence(self):
        failures = check_date_grounding("Event on 2019-01-01.", self.dates)
        self.assertEqual(len(failures), 1)
        self.assertIn("2019-01-01", failures[0])

    def test_passes_dates_present_in_evidence(self):
        self.assertEqual(
            check_date_grounding("Filed 2026-05-12 and 2026-07-08.", self.dates), []
        )

    def test_flags_amount_absent_from_evidence(self):
        failures = check_amount_grounding("Raised $999M.", self.amounts)
        self.assertEqual(len(failures), 1)
        self.assertIn("999", failures[0])

    def test_billions_and_millions_compare_equal(self):
        self.assertEqual(_normalize_amount("1.2", "billion"), 1200.0)
        # $1.2B must match evidence recorded as 1200.0 million
        self.assertEqual(check_amount_grounding("Raised $1.2B.", self.amounts), [])


class TestUnknownHonestyGuard(unittest.TestCase):
    """Regression tests for the bug where a dead Neo4j graph produced confident
    filler instead of an honest UNKNOWN."""

    def test_detects_dimension_with_no_evidence(self):
        confidence = _confidence(
            competitive={"level": "unknown", "evidence_count": None}
        )
        self.assertEqual(
            _dimensions_without_evidence(confidence), ["COMPETITIVE POSITION"]
        )

    def test_detects_zero_count_as_empty(self):
        confidence = _confidence(personnel={"level": "low", "evidence_count": 0})
        self.assertIn("PERSONNEL STABILITY", _dimensions_without_evidence(confidence))

    def test_no_empty_dimensions_when_all_have_evidence(self):
        self.assertEqual(_dimensions_without_evidence(_confidence()), [])

    def test_replaces_filler_with_unknown(self):
        confidence = _confidence(
            competitive={"level": "unknown", "evidence_count": None}
        )
        out = _enforce_unknown_sections(FIXTURE_BRIEF, confidence)
        competitive = split_sections(out)["COMPETITIVE POSITION"]
        self.assertIn(UNKNOWN_LINE, competitive)
        self.assertNotIn("prominent player", competitive)

    def test_leaves_sections_with_evidence_untouched(self):
        confidence = _confidence(
            competitive={"level": "unknown", "evidence_count": None}
        )
        out = _enforce_unknown_sections(FIXTURE_BRIEF, confidence)
        for fact in ("$150M", "75 GitHub releases", "Teresa Carlson"):
            self.assertIn(fact, out)

    def test_is_a_noop_when_all_dimensions_have_evidence(self):
        out = _enforce_unknown_sections(FIXTURE_BRIEF, _confidence())
        self.assertEqual(out, FIXTURE_BRIEF)

    def test_does_not_rewrite_an_existing_unknown(self):
        brief = FIXTURE_BRIEF.replace(
            "Acme is a prominent player in the AI development space.",
            "UNKNOWN — the competitive graph returned no rows.",
        )
        confidence = _confidence(
            competitive={"level": "unknown", "evidence_count": None}
        )
        out = _enforce_unknown_sections(brief, confidence)
        self.assertIn("returned no rows", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
