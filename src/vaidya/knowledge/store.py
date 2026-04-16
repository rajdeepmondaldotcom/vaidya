"""ChromaDB-backed vector store for government scheme retrieval."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from vaidya.models.scheme import SchemeRecord

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "schemes"


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
        self._collection: Collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def index_scheme(self, scheme: SchemeRecord) -> None:
        """Add or update a scheme in the collection."""
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
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB search failed", extra={"error": str(exc)})
            return []

        return self._results_to_records(results)

    def get_scheme(self, scheme_id: str) -> SchemeRecord | None:
        """Retrieve a single scheme by its ID. Returns ``None`` if not found."""
        try:
            result = self._collection.get(
                ids=[scheme_id],
                include=["metadatas", "documents"],
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
                include=["metadatas", "documents"],
            )
            return self._results_to_records(result)
        except Exception as exc:
            logger.error("Scheme listing failed", extra={"error": str(exc)})
            return []

    @staticmethod
    def _normalize_chromadb_field(
        raw: list | None,
        default: list | None = None,
    ) -> list:
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
            jurisdiction=meta.get("jurisdiction", "central"),
            state_code=meta.get("state_code"),
            income_thresholds=[],
            secc_categories=[],
            occupation_included=[],
            occupation_excluded=[],
            exclusion_rules=[],
            family_criteria={"family_definition": "", "head_of_family_required": False},
            geographic_restrictions=[],
            coverage_amount_inr=int(meta.get("coverage_amount_inr", 0)),
            coverage_type="per_family_per_year",
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
        return self._collection.count()

    def is_healthy(self) -> bool:
        """Check ChromaDB connectivity via heartbeat."""
        try:
            self._client.heartbeat()
            return True
        except Exception as exc:
            logger.error("ChromaDB heartbeat failed", extra={"error": str(exc)})
            return False
