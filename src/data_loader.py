"""Data loading functions for World Happiness Report data."""

from pathlib import Path

import pandas as pd

from .data_normalizer import enrich_with_continents, load_and_combine_years
from .utils import load_column_normalization, load_simple_mapping


def load_processed_data(data_dir: Path = Path("processed_data")) -> pd.DataFrame:
    """Load pre-processed happiness data from CSV.

    Args:
        data_dir: Directory containing the processed data file.

    Returns:
        DataFrame with combined happiness data from 2015-2019.
    """
    filepath = data_dir / "happiness_combined_2015_2019.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data file not found at {filepath}. Please run process_and_save_data() first."
        )
    return pd.read_csv(filepath)


def process_and_save_data(
    data_dir: Path = Path("raw_data"),
    metadata_dir: Path = Path("metadata"),
    output_dir: Path = Path("processed_data"),
) -> pd.DataFrame:
    """Process raw happiness data and save to CSV.

    Args:
        data_dir: Directory containing raw CSV files for each year.
        metadata_dir: Directory containing metadata CSV files.
        output_dir: Directory where processed data will be saved.

    Returns:
        DataFrame with combined and processed happiness data.
    """
    # Define year files
    year_files = {
        2015: data_dir / "2015.csv",
        2016: data_dir / "2016.csv",
        2017: data_dir / "2017.csv",
        2018: data_dir / "2018.csv",
        2019: data_dir / "2019.csv",
    }

    # Load metadata mappings
    column_normalization = load_column_normalization(metadata_dir / "column_normalization.csv")
    country_aliases = load_simple_mapping(metadata_dir / "country_aliases.csv", "original", "alias")
    region_overrides = load_simple_mapping(metadata_dir / "region_overrides.csv", "country", "region")
    continent_overrides = load_simple_mapping(metadata_dir / "continent_overrides.csv", "country", "continent")
    region_to_continent = load_simple_mapping(metadata_dir / "region_to_continent.csv", "region", "continent")

    # Process data
    df = load_and_combine_years(year_files, column_normalization, country_aliases, region_overrides)
    df = enrich_with_continents(df, region_to_continent, continent_overrides)

    # Save processed data
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "happiness_combined_2015_2019.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")

    return df
