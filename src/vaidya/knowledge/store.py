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

    def __init__(self, chromadb_path: str) -> None:
        self._client = chromadb.PersistentClient(path=chromadb_path)
        self._collection: Collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "KnowledgeStore initialised",
            extra={"path": chromadb_path, "collection": _COLLECTION_NAME},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def index_scheme(self, scheme: SchemeRecord) -> None:
        """Add or update a scheme in the collection.

        The ``description_for_embedding`` text is used as the document body
        for ChromaDB's built-in embedding function.  Metadata fields enable
        filtered retrieval.
        """
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

    # ------------------------------------------------------------------
    # Read -- vector search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        state_code: str | None = None,
    ) -> list[SchemeRecord]:
        """Semantic search over scheme descriptions.

        Parameters
        ----------
        query:
            Free-text query (user's health need, situation description, etc.).
        n_results:
            Maximum number of results to return.
        state_code:
            Optional ISO-style state code (e.g. ``"MH"``, ``"TN"``) to restrict
            results to a single state plus central schemes.

        Returns
        -------
        list[SchemeRecord]
            Matching schemes ordered by relevance. Empty list on error.
        """
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

    # ------------------------------------------------------------------
    # Read -- direct lookup
    # ------------------------------------------------------------------

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
        """List all indexed schemes, optionally filtered by state.

        Note: for large collections this fetches everything -- suitable for
        Phase 1 scale (< 500 schemes).
        """
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _results_to_records(
        results: dict[str, Any] | Any,
    ) -> list[SchemeRecord]:
        """Convert ChromaDB result dicts back into SchemeRecord stubs.

        ChromaDB only stores the document text and metadata we indexed -- not
        the full SchemeRecord.  The caller should join with the authoritative
        scheme registry if full records are needed.  Here we reconstruct
        *partial* SchemeRecord objects from metadata for convenience in the
        pipeline.
        """
        records: list[SchemeRecord] = []

        ids: list[str] = (
            results.get("ids", [[]])[0]
            if isinstance(results.get("ids", [[]])[0], list)
            else results.get("ids", [])
        )
        metadatas_raw = results.get("metadatas", [[]])
        metadatas: list[dict[str, Any]] = (
            metadatas_raw[0]
            if metadatas_raw and isinstance(metadatas_raw[0], list)
            else metadatas_raw or []
        )
        documents_raw = results.get("documents", [[]])
        documents: list[str] = (
            documents_raw[0]
            if documents_raw and isinstance(documents_raw[0], list)
            else documents_raw or []
        )

        for idx, scheme_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            doc = documents[idx] if idx < len(documents) else ""

            if not meta:
                continue

            try:
                # Reconstruct a minimal SchemeRecord from stored metadata.
                # Full records should be looked up from the scheme registry.
                record = SchemeRecord(
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
                    keywords=str(meta.get("keywords", "")).split(",")
                    if meta.get("keywords")
                    else [],
                )
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
