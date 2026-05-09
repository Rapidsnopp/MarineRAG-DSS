"""
Process growth threshold data from Excel files and create embeddings for RAG.

This module:
1. Reads growth threshold Excel files for aquaculture species
2. Extracts and normalizes threshold data
3. Creates embeddings for semantic search
4. Stores processed data for RAG integration
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GrowthThresholdProcessor:
    """Process growth threshold data from Excel files."""

    # Species mapping
    SPECIES_MAPPING = {
        "CÂY BẦN CHUA": {
            "name": "Cây bần chua",
            "english": "Mangrove worm",
            "code": "tree_worm",
            "type": "tree",
        },
        "HÀU": {
            "name": "Hàu",
            "english": "Oyster",
            "code": "oyster",
            "type": "shellfish",
        },
        "CÁ GIÒ": {
            "name": "Cá giò",
            "english": "Mullet",
            "code": "mullet",
            "type": "fish",
        },
    }

    THRESHOLD_LEVELS = {
        "very_unsuitable": {
            "vi": "Rất không phù hợp",
            "weight": 0.0,
            "rank": 0,
        },
        "suitable": {
            "vi": "Phù hợp",
            "weight": 0.75,
            "rank": 1,
        },
        "very_suitable": {
            "vi": "Rất phù hợp",
            "weight": 1.0,
            "rank": 2,
        },
    }

    def __init__(self, data_dir: str = "data/Growth_threshold"):
        """Initialize processor.

        Args:
            data_dir: Path to directory containing threshold Excel files
        """
        self.data_dir = Path(data_dir)
        self.processed_data: dict[str, list[dict[str, Any]]] = {}

    def read_excel_files(self) -> dict[str, pd.DataFrame]:
        """Read all Excel files from data directory.

        Returns:
            Dictionary mapping species to DataFrames
        """
        excel_files = {
            "CÂY BẦN CHUA": self.data_dir
            / "1. Chuyên gia thống nhất_CÂY BẦN CHUA.xlsx",
            "HÀU": self.data_dir / "2. Chuyên gia thống nhất_HÀU.xlsx",
            "CÁ GIÒ": self.data_dir / "3.Chuyên gia thống nhất_CÁ GIÒ.xlsx",
        }

        data = {}
        for species_name, file_path in excel_files.items():
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            try:
                df = pd.read_excel(file_path, sheet_name="3 Ngưỡng")
                data[species_name] = df
                logger.info(f"Loaded {species_name}: {df.shape} (rows x cols)")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        return data

    def _clean_threshold_value(self, value: Any) -> str:
        """Clean and normalize threshold value."""
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _extract_thresholds_from_row(self, row: pd.Series) -> dict[str, str]:
        """Extract threshold values for each level from a row.

        Returns:
            Dictionary with threshold levels as keys and values as strings
        """
        thresholds = {}
        # Columns are typically: STT, Thông số, Đơn vị, [unsuitable], [suitable], [very suitable], Rank
        if len(row) >= 6:
            thresholds["very_unsuitable"] = self._clean_threshold_value(row.iloc[3])
            thresholds["suitable"] = self._clean_threshold_value(row.iloc[4])
            thresholds["very_suitable"] = self._clean_threshold_value(row.iloc[5])
        return thresholds

    def parse_threshold_data(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, list[dict[str, Any]]]:
        """Parse threshold data from DataFrames.

        Args:
            dataframes: Dictionary of species to DataFrames

        Returns:
            Processed threshold data
        """
        processed = {}

        for species_name, df in dataframes.items():
            species_info = self.SPECIES_MAPPING.get(species_name, {})
            thresholds_list = []

            # Skip header rows and process data rows
            # Typically start from row 5 onwards
            for idx, row in df.iterrows():
                # Get parameter name and unit (usually in columns 1-2)
                param_name = None
                unit = None

                if len(row) >= 2:
                    param_name = self._clean_threshold_value(row.iloc[1])
                    unit = self._clean_threshold_value(row.iloc[2])

                if not param_name or param_name in [
                    "Thông số",
                    "STT",
                    "Gía trị trọng số",
                ]:
                    continue

                # Extract threshold values
                thresholds = self._extract_thresholds_from_row(row)

                if any(thresholds.values()):  # If any threshold value exists
                    threshold_entry = {
                        "species": species_info.get("name", species_name),
                        "species_code": species_info.get("code"),
                        "parameter": param_name,
                        "unit": unit,
                        "thresholds": {
                            "very_unsuitable": thresholds.get("very_unsuitable", ""),
                            "suitable": thresholds.get("suitable", ""),
                            "very_suitable": thresholds.get("very_suitable", ""),
                        },
                        "weights": {
                            "very_unsuitable": 0.0,
                            "suitable": 0.75,
                            "very_suitable": 1.0,
                        },
                    }
                    thresholds_list.append(threshold_entry)
                    logger.debug(
                        f"Extracted threshold: {species_name} - {param_name} ({unit})"
                    )

            processed[species_name] = thresholds_list
            logger.info(
                f"Processed {species_name}: {len(thresholds_list)} threshold parameters"
            )

        self.processed_data = processed
        return processed

    def create_documents_for_rag(self) -> list[Document]:
        """Create LangChain documents for RAG from processed data.

        Returns:
            List of Document objects for embedding and storage
        """
        documents = []

        for species_name, thresholds_list in self.processed_data.items():
            species_info = self.SPECIES_MAPPING.get(species_name, {})

            for threshold in thresholds_list:
                # Create rich text content for embedding
                content_parts = [
                    f"Growth threshold for {threshold['species']} ({species_info.get('english', '')})",
                    f"Parameter: {threshold['parameter']}",
                    f"Unit: {threshold['unit']}" if threshold['unit'] else "",
                    "",
                    "Optimal range (Very Suitable):",
                    f"  {threshold['thresholds']['very_suitable']}",
                    "",
                    "Good range (Suitable):",
                    f"  {threshold['thresholds']['suitable']}",
                    "",
                    "Poor conditions (Very Unsuitable):",
                    f"  {threshold['thresholds']['very_unsuitable']}",
                ]

                content = "\n".join(filter(None, content_parts))

                # Create metadata
                metadata = {
                    "source": f"growth_threshold_{species_name.lower().replace(' ', '_')}",
                    "species": threshold["species"],
                    "species_code": threshold["species_code"],
                    "parameter": threshold["parameter"],
                    "unit": threshold["unit"],
                    "type": "growth_threshold",
                    "thresholds_json": json.dumps(threshold["thresholds"]),
                    "weights_json": json.dumps(threshold["weights"]),
                }

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        logger.info(f"Created {len(documents)} documents for RAG")
        return documents

    def save_processed_data(
        self, output_dir: str = "data/processed_growth_thresholds"
    ) -> None:
        """Save processed data to JSON files.

        Args:
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full processed data
        full_data_path = output_path / "growth_thresholds.json"
        with open(full_data_path, "w", encoding="utf-8") as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed data to {full_data_path}")

        # Save summary for each species
        for species_name, thresholds_list in self.processed_data.items():
            species_code = self.SPECIES_MAPPING.get(species_name, {}).get("code")
            if species_code:
                species_path = output_path / f"{species_code}_thresholds.json"
                with open(species_path, "w", encoding="utf-8") as f:
                    json.dump(thresholds_list, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {species_name} data to {species_path}")

    def process_all(self) -> list[Document]:
        """Execute the full pipeline: read -> parse -> create documents.

        Returns:
            List of documents ready for RAG
        """
        logger.info("Starting growth threshold data processing...")

        # Step 1: Read Excel files
        dataframes = self.read_excel_files()
        if not dataframes:
            logger.error("No data files found!")
            return []

        # Step 2: Parse data
        self.parse_threshold_data(dataframes)

        # Step 3: Save processed data
        self.save_processed_data()

        # Step 4: Create documents for RAG
        documents = self.create_documents_for_rag()

        logger.info("Growth threshold processing completed successfully!")
        return documents


def main():
    """Main entry point for testing."""
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Process growth thresholds
    processor = GrowthThresholdProcessor()
    documents = processor.process_all()

    print(f"\nTotal documents created: {len(documents)}\n")

    # Print sample documents
    if documents:
        print("Sample document:")
        print("=" * 80)
        doc = documents[0]
        print(f"Content:\n{doc.page_content}")
        print(f"\nMetadata: {doc.metadata}")


if __name__ == "__main__":
    main()
