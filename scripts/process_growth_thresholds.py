"""
Integration script to process growth thresholds and generate embeddings for RAG.

Usage:
    python scripts/process_growth_thresholds.py [--config-path CONFIG_PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(args: argparse.Namespace) -> int:
    """Main entry point.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    setup_logging(args.log_level)

    try:
        # Import dependencies
        from src.growth_threshold_processor import GrowthThresholdProcessor
        from src.embeddings_handler import EmbeddingsHandler, create_embeddings_from_config

        logger.info("=" * 80)
        logger.info("GROWTH THRESHOLD DATA PROCESSING FOR RAG")
        logger.info("=" * 80)

        # Step 1: Process growth threshold data
        logger.info("\n[Step 1] Processing growth threshold Excel files...")
        processor = GrowthThresholdProcessor(data_dir=args.data_dir)
        documents = processor.process_all()

        if not documents:
            logger.error("No documents generated from growth thresholds!")
            return 1

        logger.info(f"✓ Generated {len(documents)} documents")

        # Step 2: Create embeddings
        logger.info("\n[Step 2] Creating embeddings...")
        embeddings_provider = create_embeddings_from_config()

        if not embeddings_provider:
            logger.error("Could not initialize embeddings provider!")
            return 1

        handler = EmbeddingsHandler(embeddings_provider)
        result = handler.embed_documents(documents)

        if not result.get("success"):
            logger.error(f"Failed to create embeddings: {result.get('error')}")
            return 1

        logger.info(
            f"✓ Created embeddings (dimension: {result['embedding_dimension']})"
        )

        # Step 3: Save embeddings
        logger.info("\n[Step 3] Saving embeddings...")
        handler.save_embeddings(output_dir=args.embeddings_dir)
        logger.info(f"✓ Embeddings saved to {args.embeddings_dir}")

        # Step 4: Generate summary report
        logger.info("\n[Step 4] Generating summary report...")
        stats = handler.get_embedding_stats()

        summary = {
            "timestamp": "",
            "total_documents": len(documents),
            "embedding_stats": stats,
            "species_breakdown": {},
            "parameters_by_species": {},
        }

        # Analyze documents
        for doc in documents:
            species = doc.metadata.get("species", "Unknown")
            param = doc.metadata.get("parameter", "Unknown")

            if species not in summary["species_breakdown"]:
                summary["species_breakdown"][species] = 0
                summary["parameters_by_species"][species] = []

            summary["species_breakdown"][species] += 1
            if param not in summary["parameters_by_species"][species]:
                summary["parameters_by_species"][species].append(param)

        # Save summary
        summary_file = Path(args.embeddings_dir) / "processing_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Summary saved to {summary_file}")

        # Print report
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nDocuments Created: {summary['total_documents']}")
        logger.info(f"Embedding Dimension: {stats['embedding_dimension']}")
        logger.info(f"Storage Size: {stats['storage_size_kb']:.2f} KB")

        logger.info("\nSpecies Breakdown:")
        for species, count in summary["species_breakdown"].items():
            params = summary["parameters_by_species"].get(species, [])
            logger.info(f"  - {species}: {count} thresholds ({len(params)} parameters)")
            for param in sorted(params)[:5]:
                logger.info(f"      • {param}")
            if len(params) > 5:
                logger.info(f"      ... and {len(params) - 5} more")

        logger.info("\n✓ All steps completed successfully!")
        logger.info(
            f"✓ Embeddings ready for RAG integration at: {args.embeddings_dir}"
        )

        return 0

    except Exception as e:
        logger.exception("An error occurred during processing")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser.

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Process growth threshold data and generate embeddings for RAG"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/Growth_threshold",
        help="Path to growth threshold data directory",
    )

    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/embeddings",
        help="Path to save embeddings",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    sys.exit(main(args))
