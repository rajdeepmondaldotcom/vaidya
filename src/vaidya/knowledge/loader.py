"""Load scheme JSON files into the ChromaDB knowledge store at startup."""

from __future__ import annotations

import logging

from vaidya.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


def load_schemes_into_store(store: KnowledgeStore) -> int:
    """Index all registered schemes into the knowledge store.

    Imports the scheme registry lazily (so the module can be tested
    independently) and calls :meth:`KnowledgeStore.index_scheme` for each.

    Returns
    -------
    int
        Number of schemes successfully indexed.
    """
    try:
        from vaidya.schemes.registry import get_schemes
    except ImportError:
        logger.warning(
            "vaidya.schemes.registry not found -- no schemes to load. "
            "Create src/vaidya/schemes/registry.py with a get_schemes() function."
        )
        return 0

    schemes = get_schemes()
    if not schemes:
        logger.warning("Scheme registry returned zero schemes")
        return 0

    indexed = 0
    for scheme in schemes:
        try:
            store.index_scheme(scheme)
            indexed += 1
        except Exception as exc:
            logger.error(
                "Failed to index scheme",
                extra={"scheme_id": scheme.scheme_id, "error": str(exc)},
            )

    logger.info(
        "Scheme loading complete",
        extra={"total_in_registry": len(schemes), "indexed": indexed},
    )
    return indexed
