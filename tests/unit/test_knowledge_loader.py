"""Tests for load_schemes_into_store (knowledge loader).

Covers:
- Loads all schemes from registry and indexes them
- Returns correct count
- Handles individual scheme indexing failure (continues with others)
- Handles missing registry module gracefully
- Handles empty registry

The KnowledgeStore is mocked to avoid needing a real ChromaDB instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaidya.knowledge.loader import load_schemes_into_store
from vaidya.models.scheme import (
    ConfidenceLevel,
    FamilyCriteria,
    Jurisdiction,
    SchemeRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheme(scheme_id: str) -> SchemeRecord:
    """Build a minimal SchemeRecord for test purposes."""
    return SchemeRecord(
        scheme_id=scheme_id,
        canonical_name=f"Scheme {scheme_id}",
        aliases=[],
        local_names={},
        jurisdiction=Jurisdiction.CENTRAL,
        state_code=None,
        income_thresholds=[],
        secc_categories=[],
        occupation_included=[],
        occupation_excluded=[],
        exclusion_rules=[],
        family_criteria=FamilyCriteria(
            family_definition="Nuclear family",
            head_of_family_required=False,
        ),
        geographic_restrictions=[],
        coverage_amount_inr=500000,
        coverage_type="per_family_per_year",
        covered_procedures=[],
        excluded_procedures=[],
        required_documents=[],
        enrollment_channels=[],
        enrollment_steps=[],
        processing_time_days=30,
        version="1.0",
        effective_date="2025-01-01",
        last_verified="2025-01-01",
        source_url="https://example.com",
        confidence_level=ConfidenceLevel.VERIFIED,
        description_for_embedding=f"Description for {scheme_id}",
        keywords=["test"],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store() -> MagicMock:
    """A mock KnowledgeStore that tracks index_scheme calls."""
    store = MagicMock()
    store.index_scheme = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadSchemesIntoStore:
    """Tests for load_schemes_into_store.

    The loader does a lazy ``from vaidya.schemes.registry import get_schemes``
    inside the function body, so we patch ``vaidya.schemes.registry.get_schemes``
    (the canonical location) rather than an attribute on the loader module.
    """

    def test_loads_all_schemes_and_returns_count(self, mock_store: MagicMock) -> None:
        schemes = [_make_scheme("S1"), _make_scheme("S2"), _make_scheme("S3")]

        with patch("vaidya.schemes.registry.get_schemes", return_value=schemes):
            count = load_schemes_into_store(mock_store)

        assert count == 3
        assert mock_store.index_scheme.call_count == 3

    def test_returns_zero_when_registry_empty(self, mock_store: MagicMock) -> None:
        with patch("vaidya.schemes.registry.get_schemes", return_value=[]):
            count = load_schemes_into_store(mock_store)

        assert count == 0
        mock_store.index_scheme.assert_not_called()

    def test_continues_on_individual_indexing_failure(self, mock_store: MagicMock) -> None:
        schemes = [_make_scheme("OK1"), _make_scheme("FAIL"), _make_scheme("OK2")]

        # Fail on the second scheme only
        def side_effect(scheme: SchemeRecord) -> None:
            if scheme.scheme_id == "FAIL":
                raise RuntimeError("ChromaDB write error")

        mock_store.index_scheme.side_effect = side_effect

        with patch("vaidya.schemes.registry.get_schemes", return_value=schemes):
            count = load_schemes_into_store(mock_store)

        # 2 succeeded, 1 failed
        assert count == 2
        assert mock_store.index_scheme.call_count == 3

    def test_handles_missing_registry_module(self, mock_store: MagicMock) -> None:
        """When vaidya.schemes.registry is not importable, returns 0.

        We simulate this by making the lazy import inside the function
        raise ImportError. Since the function does
        ``from vaidya.schemes.registry import get_schemes`` in a try/except,
        we patch builtins.__import__ to raise for that specific module.
        """
        import builtins

        real_import = builtins.__import__

        def _failing_import(name: str, *args, **kwargs):
            if name == "vaidya.schemes.registry":
                raise ImportError("simulated missing module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_failing_import):
            count = load_schemes_into_store(mock_store)

        assert count == 0
        mock_store.index_scheme.assert_not_called()

    def test_indexes_each_scheme_with_correct_object(self, mock_store: MagicMock) -> None:
        s1 = _make_scheme("A")
        s2 = _make_scheme("B")

        with patch("vaidya.schemes.registry.get_schemes", return_value=[s1, s2]):
            load_schemes_into_store(mock_store)

        calls = mock_store.index_scheme.call_args_list
        assert calls[0][0][0].scheme_id == "A"
        assert calls[1][0][0].scheme_id == "B"
