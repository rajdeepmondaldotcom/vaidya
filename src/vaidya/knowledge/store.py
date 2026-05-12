"""ChromaDB-backed vector store for government scheme retrieval."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from vaidya.models.scheme import FamilyCriteria, Jurisdiction, SchemeCoverageType, SchemeRecord

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "schemes"


class _InMemorySchemeStore:
    """Small fallback index used when ChromaDB cannot start locally."""

    def __init__(self) -> None:
        self._records: dict[str, SchemeRecord] = {}

    def index_scheme(self, scheme: SchemeRecord) -> None:
        self._records[scheme.scheme_id] = scheme

    def search(
        self,
        query: str,
        n_results: int = 5,
        state_code: str | None = None,
    ) -> list[SchemeRecord]:
        query_terms = set(query.lower().replace("/", " ").split())
        ranked: list[tuple[int, str, SchemeRecord]] = []
        for scheme in self.list_schemes(state_code):
            haystack = " ".join(
                [
                    scheme.description_for_embedding,
                    scheme.canonical_name,
                    " ".join(scheme.keywords),
                ]
            ).lower()
            score = sum(1 for term in query_terms if term in haystack)
            ranked.append((score, scheme.scheme_id, scheme))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [scheme for _, _, scheme in ranked[:n_results]]

    def get_scheme(self, scheme_id: str) -> SchemeRecord | None:
        return self._records.get(scheme_id)

    def list_schemes(self, state_code: str | None = None) -> list[SchemeRecord]:
        records = list(self._records.values())
        if not state_code:
            return records
        return [
            scheme
            for scheme in records
            if scheme.state_code == state_code or scheme.jurisdiction == Jurisdiction.CENTRAL
        ]

    @property
    def count(self) -> int:
        return len(self._records)


class KnowledgeStore:
    """Thin wrapper around a ChromaDB persistent collection.

    Stores :class:`SchemeRecord` objects indexed by their
    ``description_for_embedding`` field. Supports vector search with optional
    state-code filtering and direct ID lookup.
    """

    def __init__(
        self,
        chromadb_path: str = "./chroma_data",
        chromadb_host: str = "",
        chromadb_port: int = 8000,
    ) -> None:
        self._client: Any | None = None
        self._collection: Collection | None = None
        self._fallback: _InMemorySchemeStore | None = None

        try:
            if chromadb_host:
                self._client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
                logger.info(
                    "KnowledgeStore connected via HTTP",
                    extra={"host": chromadb_host, "port": chromadb_port},
                )
            else:
                self._client = chromadb.PersistentClient(path=chromadb_path)
                logger.info(
                    "KnowledgeStore using persistent storage",
                    extra={"path": chromadb_path},
                )
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.error(
                "ChromaDB unavailable; falling back to in-memory scheme search",
                extra={"error": str(exc), "path": chromadb_path, "host": chromadb_host},
            )
            self._fallback = _InMemorySchemeStore()

    def index_scheme(self, scheme: SchemeRecord) -> None:
        """Add or update a scheme in the collection."""
        if self._fallback is not None:
            self._fallback.index_scheme(scheme)
            return
        if self._collection is None:
            raise RuntimeError("KnowledgeStore is not initialised")

        metadata: dict[str, Any] = {
            "scheme_id": scheme.scheme_id,
            "jurisdiction": scheme.jurisdiction.value,
            "keywords": ",".join(scheme.keywords),
            "canonical_name": scheme.canonical_name,
            "coverage_amount_inr": scheme.coverage_amount_inr,
            "confidence_level": scheme.confidence_level.value,
        }
        if scheme.state_code:
            metadata["state_code"] = scheme.state_code

        self._collection.upsert(
            ids=[scheme.scheme_id],
            documents=[scheme.description_for_embedding],
            metadatas=[metadata],
        )
        logger.debug(
            "Scheme indexed",
            extra={"scheme_id": scheme.scheme_id, "name": scheme.canonical_name},
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        state_code: str | None = None,
    ) -> list[SchemeRecord]:
        """Semantic search over scheme descriptions, optionally filtered by state."""
        if self._fallback is not None:
            return self._fallback.search(query, n_results=n_results, state_code=state_code)
        if self._collection is None:
            return []

        where_filter: dict[str, Any] | None = None
        if state_code:
            # Return central schemes AND the target state's schemes
            where_filter = {
                "$or": [
                    {"state_code": state_code},
                    {"jurisdiction": "central"},
                ]
            }

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["metadatas", "documents", "distances"],  # type: ignore[list-item]
            )
        except Exception as exc:
            logger.error("ChromaDB search failed", extra={"error": str(exc)})
            return []

        return self._results_to_records(results)

    def get_scheme(self, scheme_id: str) -> SchemeRecord | None:
        """Retrieve a single scheme by its ID. Returns ``None`` if not found."""
        if self._fallback is not None:
            return self._fallback.get_scheme(scheme_id)
        if self._collection is None:
            return None

        try:
            result = self._collection.get(
                ids=[scheme_id],
                include=["metadatas", "documents"],  # type: ignore[list-item]
            )
            records = self._results_to_records(result)
            return records[0] if records else None
        except Exception as exc:
            logger.error(
                "Scheme lookup failed",
                extra={"scheme_id": scheme_id, "error": str(exc)},
            )
            return None

    def list_schemes(self, state_code: str | None = None) -> list[SchemeRecord]:
        """List all indexed schemes, optionally filtered by state."""
        if self._fallback is not None:
            return self._fallback.list_schemes(state_code)
        if self._collection is None:
            return []

        where_filter: dict[str, Any] | None = None
        if state_code:
            where_filter = {
                "$or": [
                    {"state_code": state_code},
                    {"jurisdiction": "central"},
                ]
            }

        try:
            result = self._collection.get(
                where=where_filter,
                include=["metadatas", "documents"],  # type: ignore[list-item]
            )
            return self._results_to_records(result)
        except Exception as exc:
            logger.error("Scheme listing failed", extra={"error": str(exc)})
            return []

    @staticmethod
    def _normalize_chromadb_field(
        raw: list[Any] | None,
        default: list[Any] | None = None,
    ) -> list[Any]:
        """Unpack ChromaDB's inconsistent nesting (list-of-lists vs flat list)."""
        if default is None:
            default = []
        if not raw:
            return default
        if raw and isinstance(raw[0], list):
            return raw[0]
        return raw

    @staticmethod
    def _metadata_to_record(
        scheme_id: str,
        meta: dict[str, Any],
        doc: str,
    ) -> SchemeRecord:
        """Reconstruct a minimal SchemeRecord stub from ChromaDB metadata."""
        return SchemeRecord(
            scheme_id=str(scheme_id),
            canonical_name=str(meta.get("canonical_name", "")),
            aliases=[],
            local_names={},
            jurisdiction=Jurisdiction(str(meta.get("jurisdiction", "central"))),
            state_code=meta.get("state_code"),
            income_thresholds=[],
            secc_categories=[],
            occupation_included=[],
            occupation_excluded=[],
            exclusion_rules=[],
            family_criteria=FamilyCriteria(
                family_definition="",
                head_of_family_required=False,
            ),
            geographic_restrictions=[],
            coverage_amount_inr=int(meta.get("coverage_amount_inr", 0)),
            coverage_type=SchemeCoverageType.PER_FAMILY_PER_YEAR,
            covered_procedures=[],
            excluded_procedures=[],
            required_documents=[],
            enrollment_channels=[],
            enrollment_steps=[],
            processing_time_days=0,
            version="stub",
            effective_date="",
            last_verified="",
            source_url="",
            confidence_level=meta.get("confidence_level", "provisional"),
            description_for_embedding=doc or "",
            keywords=str(meta.get("keywords", "")).split(",") if meta.get("keywords") else [],
        )

    @staticmethod
    def _results_to_records(
        results: dict[str, Any] | Any,
    ) -> list[SchemeRecord]:
        """Convert ChromaDB result dicts back into SchemeRecord stubs."""
        if not isinstance(results, dict):
            logger.warning("ChromaDB returned unexpected type: %s", type(results).__name__)
            return []

        records: list[SchemeRecord] = []

        ids = KnowledgeStore._normalize_chromadb_field(results.get("ids", [[]]))
        metadatas = KnowledgeStore._normalize_chromadb_field(results.get("metadatas", [[]]))
        documents = KnowledgeStore._normalize_chromadb_field(results.get("documents", [[]]))

        for idx, scheme_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            doc = documents[idx] if idx < len(documents) else ""

            if not meta:
                continue

            try:
                record = KnowledgeStore._metadata_to_record(scheme_id, meta, doc)
                records.append(record)
            except Exception as exc:
                logger.warning(
                    "Failed to reconstruct SchemeRecord from ChromaDB metadata",
                    extra={"scheme_id": scheme_id, "error": str(exc)},
                )

        return records

    @property
    def count(self) -> int:
        """Number of schemes currently indexed."""
        if self._fallback is not None:
            return self._fallback.count
        if self._collection is None:
            return 0
        return self._collection.count()

    def is_healthy(self) -> bool:
        """Check ChromaDB connectivity via heartbeat."""
        if self._fallback is not None:
            return False
        if self._client is None:
            return False
        try:
            self._client.heartbeat()
            return True
        except Exception as exc:
            logger.error("ChromaDB heartbeat failed", extra={"error": str(exc)})
            return False
