"""Load scheme data from JSON files into ChromaDB.

This script reads all scheme records from ``src/vaidya/schemes/data/*.json``,
validates them as ``SchemeRecord`` objects, and indexes them into the ChromaDB
persistent knowledge store.

Usage
-----
    python scripts/seed_knowledge.py
    python scripts/seed_knowledge.py --chromadb-path ./chroma_data
    python scripts/seed_knowledge.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project paths so this script can be run standalone
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from vaidya.knowledge.store import KnowledgeStore  # noqa: E402
from vaidya.models.scheme import SchemeRecord  # noqa: E402

logger = logging.getLogger(__name__)

_SCHEME_DATA_DIR = _SRC_DIR / "vaidya" / "schemes" / "data"
_DEFAULT_CHROMADB_PATH = str(_PROJECT_ROOT / "chroma_data")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the ChromaDB knowledge store with scheme data.",
    )
    parser.add_argument(
        "--chromadb-path",
        default=_DEFAULT_CHROMADB_PATH,
        help=f"Path to ChromaDB persistent storage (default: {_DEFAULT_CHROMADB_PATH})",
    )
    parser.add_argument(
        "--scheme-dir",
        default=str(_SCHEME_DATA_DIR),
        help=f"Directory containing scheme JSON files (default: {_SCHEME_DATA_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print scheme data without writing to ChromaDB",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def load_scheme_files(scheme_dir: Path) -> list[SchemeRecord]:
    """Load and validate all scheme JSON files from *scheme_dir*."""
    if not scheme_dir.exists():
        logger.error("Scheme data directory does not exist: %s", scheme_dir)
        return []

    json_files = sorted(scheme_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", scheme_dir)
        return []

    schemes: list[SchemeRecord] = []
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            record = SchemeRecord.model_validate(data)
            schemes.append(record)
            logger.info(
                "  Loaded: %-30s (%s)",
                record.canonical_name,
                record.scheme_id,
            )
        except json.JSONDecodeError as exc:
            logger.error("  Invalid JSON in %s: %s", json_file.name, exc)
        except Exception as exc:
            logger.error("  Validation failed for %s: %s", json_file.name, exc)

    return schemes


def seed_knowledge_store(
    schemes: list[SchemeRecord],
    chromadb_path: str,
) -> int:
    """Index all *schemes* into a ChromaDB collection at *chromadb_path*.

    Returns the number of schemes successfully indexed.
    """
    store = KnowledgeStore(chromadb_path)
    indexed = 0

    for scheme in schemes:
        try:
            store.index_scheme(scheme)
            indexed += 1
        except Exception as exc:
            logger.error(
                "  Failed to index %s: %s",
                scheme.scheme_id,
                exc,
            )

    return indexed


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    scheme_dir = Path(args.scheme_dir)
    print(f"\nScheme data directory: {scheme_dir}")
    print(f"ChromaDB path:        {args.chromadb_path}")
    print(f"Dry run:              {args.dry_run}\n")

    # Load and validate
    print("Loading scheme files...")
    schemes = load_scheme_files(scheme_dir)

    if not schemes:
        print("\nNo schemes loaded. Check the scheme data directory.")
        sys.exit(1)

    print(f"\nValidated {len(schemes)} scheme(s):\n")
    for s in schemes:
        state_info = f"  [{s.state_code}]" if s.state_code else "  [central]"
        coverage = f"Rs {s.coverage_amount_inr:,}" if s.coverage_amount_inr else "comprehensive"
        print(f"  {s.scheme_id:<25} {state_info:<12} {s.canonical_name}")
        print(f"  {'':25} {'':12} Coverage: {coverage} | Confidence: {s.confidence_level.value}")
        print()

    if args.dry_run:
        print("Dry run complete. No data written to ChromaDB.")
        return

    # Seed
    print(f"Indexing into ChromaDB at {args.chromadb_path}...")
    indexed = seed_knowledge_store(schemes, args.chromadb_path)

    print(f"\nDone. Indexed {indexed}/{len(schemes)} scheme(s) into ChromaDB.")

    if indexed < len(schemes):
        print(f"WARNING: {len(schemes) - indexed} scheme(s) failed to index.")
        sys.exit(1)

    # Verify
    store = KnowledgeStore(args.chromadb_path)
    print(f"Verification: {store.count} scheme(s) in the collection.")


if __name__ == "__main__":
    main()
