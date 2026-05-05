"""Tests for ChromaDB-backed KnowledgeStore.

Uses an in-memory fake ChromaDB client injected into the KnowledgeStore to
avoid filesystem, SQLite extension, or network dependencies in unit tests.

Covers:
- index_scheme: upserts a scheme into the collection
- search: returns relevant schemes, supports state_code filter
- get_scheme: lookup by ID, returns None for unknown
- list_schemes: returns all indexed schemes
- count: correct number of indexed schemes
- is_healthy: returns True when ChromaDB is accessible
- Error handling: search returns empty list on error
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaidya.knowledge.store import KnowledgeStore
from vaidya.models.scheme import (
    ConfidenceLevel,
    FamilyCriteria,
    Jurisdiction,
    SchemeRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheme(
    scheme_id: str = "TEST-001",
    canonical_name: str = "Test Scheme",
    jurisdiction: Jurisdiction = Jurisdiction.CENTRAL,
    state_code: str | None = None,
    coverage_amount_inr: int = 500000,
    description: str = "A test healthcare scheme for below poverty line families",
    keywords: list[str] | None = None,
    confidence_level: ConfidenceLevel = ConfidenceLevel.VERIFIED,
) -> SchemeRecord:
    """Build a minimal valid SchemeRecord for testing."""
    return SchemeRecord(
        scheme_id=scheme_id,
        canonical_name=canonical_name,
        aliases=[],
        local_names={},
        jurisdiction=jurisdiction,
        state_code=state_code,
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
        coverage_amount_inr=coverage_amount_inr,
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
        confidence_level=confidence_level,
        description_for_embedding=description,
        keywords=keywords or ["healthcare", "government"],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeCollection:
    """Minimal Chroma collection stand-in for deterministic unit tests."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for scheme_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self._items[scheme_id] = {"document": document, "metadata": metadata}

    def count(self) -> int:
        return len(self._items)

    def query(
        self,
        *,
        query_texts: list[str],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str],
    ) -> dict[str, list[list[Any]]]:
        del include
        query_terms = set(query_texts[0].lower().replace("/", " ").split())

        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for scheme_id, item in self._items.items():
            if not self._matches_where(item["metadata"], where):
                continue
            haystack = " ".join(
                [
                    item["document"],
                    str(item["metadata"].get("canonical_name", "")),
                    str(item["metadata"].get("keywords", "")).replace(",", " "),
                ]
            ).lower()
            score = sum(1 for term in query_terms if term in haystack)
            ranked.append((score, scheme_id, item))

        ranked.sort(key=lambda row: (-row[0], row[1]))
        selected = ranked[:n_results]
        return {
            "ids": [[scheme_id for _, scheme_id, _ in selected]],
            "metadatas": [[item["metadata"] for _, _, item in selected]],
            "documents": [[item["document"] for _, _, item in selected]],
        }

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        include: list[str],
    ) -> dict[str, list[Any]]:
        del include
        selected: list[tuple[str, dict[str, Any]]] = []
        if ids is not None:
            for scheme_id in ids:
                item = self._items.get(scheme_id)
                if item is not None and self._matches_where(item["metadata"], where):
                    selected.append((scheme_id, item))
        else:
            selected = [
                (scheme_id, item)
                for scheme_id, item in self._items.items()
                if self._matches_where(item["metadata"], where)
            ]

        return {
            "ids": [scheme_id for scheme_id, _ in selected],
            "metadatas": [item["metadata"] for _, item in selected],
            "documents": [item["document"] for _, item in selected],
        }

    @classmethod
    def _matches_where(cls, metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if where is None:
            return True
        if "$or" in where:
            return any(cls._matches_where(metadata, condition) for condition in where["$or"])
        return all(metadata.get(key) == value for key, value in where.items())


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> _FakeCollection:
        del name, metadata
        return self.collection

    def heartbeat(self) -> int:
        return 1


@pytest.fixture()
def store() -> KnowledgeStore:
    """KnowledgeStore backed by an in-memory fake ChromaDB client."""
    fake_client = _FakeChromaClient()

    with patch(
        "vaidya.knowledge.store.chromadb.PersistentClient",
        return_value=fake_client,
    ):
        ks = KnowledgeStore(chromadb_path="/tmp/test_chroma_unused")

    return ks


@pytest.fixture()
def populated_store(store: KnowledgeStore) -> KnowledgeStore:
    """A store pre-loaded with several schemes for search tests."""
    schemes = [
        _make_scheme(
            scheme_id="PMJAY",
            canonical_name="Ayushman Bharat PM-JAY",
            description="Free health insurance for BPL families up to 5 lakh",
            keywords=["pmjay", "hospitalization", "bpl"],
        ),
        _make_scheme(
            scheme_id="CMCHIS",
            canonical_name="CM Comprehensive Health Insurance",
            jurisdiction=Jurisdiction.STATE,
            state_code="TN",
            description="Tamil Nadu state health insurance for BPL families",
            keywords=["tamil_nadu", "health_insurance"],
        ),
        _make_scheme(
            scheme_id="BSBY",
            canonical_name="Bhamashah Swasthya Bima Yojana",
            jurisdiction=Jurisdiction.STATE,
            state_code="RJ",
            description="Rajasthan health insurance for BPL/NFSA families",
            keywords=["rajasthan", "cashless", "bpl"],
        ),
        _make_scheme(
            scheme_id="JSSK",
            canonical_name="Janani Shishu Suraksha Karyakram",
            description="Free maternity and infant care for pregnant women",
            keywords=["maternity", "infant", "pregnancy"],
        ),
    ]
    for s in schemes:
        store.index_scheme(s)
    return store


# ---------------------------------------------------------------------------
# index_scheme
# ---------------------------------------------------------------------------


class TestIndexScheme:
    def test_upserts_scheme_into_collection(self, store: KnowledgeStore) -> None:
        scheme = _make_scheme(scheme_id="IDX-001", canonical_name="Index Test")
        store.index_scheme(scheme)

        assert store.count == 1

    def test_upsert_replaces_existing(self, store: KnowledgeStore) -> None:
        scheme_v1 = _make_scheme(
            scheme_id="IDX-002",
            canonical_name="Version 1",
            description="Original description",
        )
        store.index_scheme(scheme_v1)

        scheme_v2 = _make_scheme(
            scheme_id="IDX-002",
            canonical_name="Version 2",
            description="Updated description",
        )
        store.index_scheme(scheme_v2)

        # Should still be 1 because it's an upsert, not an insert
        assert store.count == 1
        record = store.get_scheme("IDX-002")
        assert record is not None
        assert record.canonical_name == "Version 2"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_relevant_schemes(self, populated_store: KnowledgeStore) -> None:
        results = populated_store.search("health insurance for poor families")

        assert len(results) > 0
        ids = {r.scheme_id for r in results}
        # PMJAY is highly relevant to "health insurance for poor families"
        assert "PMJAY" in ids

    def test_search_with_state_code_filter(self, populated_store: KnowledgeStore) -> None:
        results = populated_store.search("health insurance", state_code="TN")

        ids = {r.scheme_id for r in results}
        # Should include central schemes and TN state scheme
        # Should NOT include RJ-specific scheme
        assert "BSBY" not in ids

    def test_search_returns_empty_on_error(self, store: KnowledgeStore) -> None:
        # Force an error by breaking the collection's query method
        store._collection.query = MagicMock(side_effect=RuntimeError("ChromaDB error"))

        results = store.search("anything")
        assert results == []

    def test_search_respects_n_results(self, populated_store: KnowledgeStore) -> None:
        results = populated_store.search("healthcare", n_results=2)

        assert len(results) <= 2


# ---------------------------------------------------------------------------
# get_scheme
# ---------------------------------------------------------------------------


class TestGetScheme:
    def test_returns_scheme_by_id(self, populated_store: KnowledgeStore) -> None:
        record = populated_store.get_scheme("PMJAY")

        assert record is not None
        assert record.scheme_id == "PMJAY"
        assert record.canonical_name == "Ayushman Bharat PM-JAY"

    def test_returns_none_for_unknown_id(self, populated_store: KnowledgeStore) -> None:
        record = populated_store.get_scheme("NONEXISTENT-999")

        assert record is None


# ---------------------------------------------------------------------------
# list_schemes
# ---------------------------------------------------------------------------


class TestListSchemes:
    def test_returns_all_indexed_schemes(self, populated_store: KnowledgeStore) -> None:
        all_schemes = populated_store.list_schemes()

        assert len(all_schemes) == 4
        ids = {s.scheme_id for s in all_schemes}
        assert ids == {"PMJAY", "CMCHIS", "BSBY", "JSSK"}


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    def test_empty_store_count_is_zero(self, store: KnowledgeStore) -> None:
        assert store.count == 0

    def test_count_after_indexing(self, store: KnowledgeStore) -> None:
        for i in range(3):
            store.index_scheme(_make_scheme(scheme_id=f"CNT-{i}", canonical_name=f"Scheme {i}"))
        assert store.count == 3


# ---------------------------------------------------------------------------
# is_healthy
# ---------------------------------------------------------------------------


class TestIsHealthy:
    def test_returns_true_when_accessible(self, store: KnowledgeStore) -> None:
        assert store.is_healthy() is True

    def test_returns_false_on_heartbeat_failure(self, store: KnowledgeStore) -> None:
        store._client.heartbeat = MagicMock(side_effect=RuntimeError("ChromaDB unreachable"))

        assert store.is_healthy() is False
