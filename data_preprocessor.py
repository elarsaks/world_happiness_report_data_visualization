"""
World Happiness Report Data Preprocessor
=========================================

This module provides utilities for loading, cleaning, and combining World Happiness
Report data from 2015-2019 into a single, analysis-ready dataset.

Data Source
-----------
Original data from the World Happiness Report, available on Kaggle:
https://www.kaggle.com/datasets/unsdsn/world-happiness

Features
--------
- Normalizes inconsistent column names across yearly datasets
- Standardizes country names for cross-year comparison
- Fills missing region data using lookups and overrides
- Enriches data with continent information
- Validates output quality with sanity checks

Usage
-----
As a module:
    >>> from data_preprocessor import load_processed_data, process_and_save_data
    >>> df = load_processed_data()  # Load existing processed data
    >>> df = process_and_save_data()  # Regenerate from raw files

From command line:
    $ python data_preprocessor.py
    $ python data_preprocessor.py --validate-only

Output Columns
--------------
- country: Standardized country name
- region: UN geoscheme-style world region (e.g., Western Europe, South Asia)
- continent: Continent (7-continent model: Africa, Asia, Europe, North America, South America, Oceania, Antarctica)
- year: Survey year (2015-2019)
- happiness_score: Overall happiness score
- gdp_per_capita: Economic output contribution
- social_support: Social support contribution
- healthy_life_expectancy: Health contribution
- freedom: Freedom to make life choices contribution
- generosity: Generosity contribution
- corruption_perception: Trust in government contribution

Author: elarsaks
License: CC0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project defaults
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_FILENAME = "happiness_combined_2015_2019.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / PROCESSED_FILENAME

# Explicit column order for reproducible output
OUTPUT_COLUMNS = [
    "country",
    "region",
    "continent",
    "year",
    "happiness_score",
    "gdp_per_capita",
    "social_support",
    "healthy_life_expectancy",
    "freedom",
    "generosity",
    "corruption_perception",
]


# Metadata embedded directly in this module to avoid external CSV dependencies.
COLUMN_NORMALIZATION: Dict[int, Dict[str, str]] = {
    2015: {
        "Country": "country",
        "Region": "region",
        "Happiness Rank": "happiness_rank",
        "Happiness Score": "happiness_score",
        "Economy (GDP per Capita)": "gdp_per_capita",
        "Family": "social_support",
        "Health (Life Expectancy)": "healthy_life_expectancy",
        "Freedom": "freedom",
        "Generosity": "generosity",
        "Trust (Government Corruption)": "corruption_perception",
        "Standard Error": "standard_error",
        "Dystopia Residual": "dystopia_residual",
    },
    2016: {
        "Country": "country",
        "Region": "region",
        "Happiness Rank": "happiness_rank",
        "Happiness Score": "happiness_score",
        "Economy (GDP per Capita)": "gdp_per_capita",
        "Family": "social_support",
        "Health (Life Expectancy)": "healthy_life_expectancy",
        "Freedom": "freedom",
        "Generosity": "generosity",
        "Trust (Government Corruption)": "corruption_perception",
        "Lower Confidence Interval": "whisker_low",
        "Upper Confidence Interval": "whisker_high",
        "Dystopia Residual": "dystopia_residual",
    },
    2017: {
        "Country": "country",
        "Happiness.Rank": "happiness_rank",
        "Happiness.Score": "happiness_score",
        "Economy..GDP.per.Capita.": "gdp_per_capita",
        "Family": "social_support",
        "Health..Life.Expectancy.": "healthy_life_expectancy",
        "Freedom": "freedom",
        "Generosity": "generosity",
        "Trust..Government.Corruption.": "corruption_perception",
        "Whisker.high": "whisker_high",
        "Whisker.low": "whisker_low",
        "Dystopia.Residual": "dystopia_residual",
    },
    2018: {
        "Country or region": "country",
        "Overall rank": "happiness_rank",
        "Score": "happiness_score",
        "GDP per capita": "gdp_per_capita",
        "Social support": "social_support",
        "Healthy life expectancy": "healthy_life_expectancy",
        "Freedom to make life choices": "freedom",
        "Generosity": "generosity",
        "Perceptions of corruption": "corruption_perception",
    },
    2019: {
        "Country or region": "country",
        "Overall rank": "happiness_rank",
        "Score": "happiness_score",
        "GDP per capita": "gdp_per_capita",
        "Social support": "social_support",
        "Healthy life expectancy": "healthy_life_expectancy",
        "Freedom to make life choices": "freedom",
        "Generosity": "generosity",
        "Perceptions of corruption": "corruption_perception",
    },
}

# =============================================================================
# GEOGRAPHIC MAPPINGS
# NOTE: The following mappings (COUNTRY_ALIASES, COUNTRY_TO_REGION, REGION_TO_CONTINENT)
# were generated with AI assistance and have not been manually verified for complete
# accuracy. Users should review and validate these mappings for their specific use cases.
# =============================================================================

COUNTRY_ALIASES: Dict[str, str] = {
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Taiwan Province of China": "Taiwan",
    "Hong Kong S.A.R., China": "Hong Kong",
    "North Macedonia": "Macedonia",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Congo, Dem. Rep.": "Congo (Kinshasa)",
    "Democratic Republic of the Congo": "Congo (Kinshasa)",
    "Congo, Rep.": "Congo (Brazzaville)",
    "Republic of the Congo": "Congo (Brazzaville)",
    "Eswatini": "Swaziland",
    "Ivory Coast": "Ivory Coast",
    "Czechia": "Czech Republic",
    "United States of America": "United States",
    "Russia": "Russian Federation",
    "Slovak Republic": "Slovakia",
}


REGION_TO_CONTINENT: Dict[str, str] = {
    "Eastern Europe": "Europe",
    "Western Europe": "Europe",
    "Northern Europe": "Europe",
    "Southern Europe": "Europe",
    "Northern Africa": "Africa",
    "Sub-Saharan Africa": "Africa",
    "Western Asia / Middle East": "Asia",
    "Central Asia": "Asia",
    "South Asia": "Asia",
    "East Asia": "Asia",
    "Southeast Asia": "Asia",
    "North America": "North America",
    "Central America": "North America",
    "Caribbean": "North America",
    "South America": "South America",
    "Oceania": "Oceania",
    "Antarctica": "Antarctica",
}


COUNTRY_TO_REGION: Dict[str, str] = {
    "Afghanistan": "South Asia",
    "Albania": "Southern Europe",
    "Algeria": "Northern Africa",
    "Angola": "Sub-Saharan Africa",
    "Argentina": "South America",
    "Armenia": "Western Asia / Middle East",
    "Australia": "Oceania",
    "Austria": "Western Europe",
    "Azerbaijan": "Western Asia / Middle East",
    "Bahrain": "Western Asia / Middle East",
    "Bangladesh": "South Asia",
    "Belarus": "Eastern Europe",
    "Belgium": "Western Europe",
    "Belize": "Central America",
    "Benin": "Sub-Saharan Africa",
    "Bhutan": "South Asia",
    "Bolivia": "South America",
    "Bosnia and Herzegovina": "Southern Europe",
    "Botswana": "Sub-Saharan Africa",
    "Brazil": "South America",
    "Bulgaria": "Eastern Europe",
    "Burkina Faso": "Sub-Saharan Africa",
    "Burundi": "Sub-Saharan Africa",
    "Cambodia": "Southeast Asia",
    "Cameroon": "Sub-Saharan Africa",
    "Canada": "North America",
    "Central African Republic": "Sub-Saharan Africa",
    "Chad": "Sub-Saharan Africa",
    "Chile": "South America",
    "China": "East Asia",
    "Colombia": "South America",
    "Comoros": "Sub-Saharan Africa",
    "Congo (Brazzaville)": "Sub-Saharan Africa",
    "Congo (Kinshasa)": "Sub-Saharan Africa",
    "Costa Rica": "Central America",
    "Croatia": "Southern Europe",
    "Cyprus": "Western Asia / Middle East",
    "Czech Republic": "Eastern Europe",
    "Denmark": "Northern Europe",
    "Djibouti": "Sub-Saharan Africa",
    "Dominican Republic": "Caribbean",
    "Ecuador": "South America",
    "Egypt": "Northern Africa",
    "El Salvador": "Central America",
    "Estonia": "Northern Europe",
    "Ethiopia": "Sub-Saharan Africa",
    "Finland": "Northern Europe",
    "France": "Western Europe",
    "Gabon": "Sub-Saharan Africa",
    "Gambia": "Sub-Saharan Africa",
    "Georgia": "Western Asia / Middle East",
    "Germany": "Western Europe",
    "Ghana": "Sub-Saharan Africa",
    "Greece": "Southern Europe",
    "Guatemala": "Central America",
    "Guinea": "Sub-Saharan Africa",
    "Haiti": "Caribbean",
    "Honduras": "Central America",
    "Hong Kong": "East Asia",
    "Hungary": "Eastern Europe",
    "Iceland": "Northern Europe",
    "India": "South Asia",
    "Indonesia": "Southeast Asia",
    "Iran": "Western Asia / Middle East",
    "Iraq": "Western Asia / Middle East",
    "Ireland": "Northern Europe",
    "Israel": "Western Asia / Middle East",
    "Italy": "Southern Europe",
    "Ivory Coast": "Sub-Saharan Africa",
    "Jamaica": "Caribbean",
    "Japan": "East Asia",
    "Jordan": "Western Asia / Middle East",
    "Kazakhstan": "Central Asia",
    "Kenya": "Sub-Saharan Africa",
    "Kosovo": "Southern Europe",
    "Kuwait": "Western Asia / Middle East",
    "Kyrgyzstan": "Central Asia",
    "Laos": "Southeast Asia",
    "Latvia": "Northern Europe",
    "Lebanon": "Western Asia / Middle East",
    "Lesotho": "Sub-Saharan Africa",
    "Liberia": "Sub-Saharan Africa",
    "Libya": "Northern Africa",
    "Lithuania": "Northern Europe",
    "Luxembourg": "Western Europe",
    "Macedonia": "Southern Europe",
    "Madagascar": "Sub-Saharan Africa",
    "Malawi": "Sub-Saharan Africa",
    "Malaysia": "Southeast Asia",
    "Mali": "Sub-Saharan Africa",
    "Malta": "Southern Europe",
    "Mauritania": "Sub-Saharan Africa",
    "Mauritius": "Sub-Saharan Africa",
    "Mexico": "Central America",
    "Moldova": "Eastern Europe",
    "Mongolia": "East Asia",
    "Montenegro": "Southern Europe",
    "Morocco": "Northern Africa",
    "Mozambique": "Sub-Saharan Africa",
    "Myanmar": "Southeast Asia",
    "Namibia": "Sub-Saharan Africa",
    "Nepal": "South Asia",
    "Netherlands": "Western Europe",
    "New Zealand": "Oceania",
    "Nicaragua": "Central America",
    "Niger": "Sub-Saharan Africa",
    "Nigeria": "Sub-Saharan Africa",
    "North Cyprus": "Western Asia / Middle East",
    "Northern Cyprus": "Western Asia / Middle East",
    "Norway": "Northern Europe",
    "Oman": "Western Asia / Middle East",
    "Pakistan": "South Asia",
    "Palestinian Territories": "Western Asia / Middle East",
    "Panama": "Central America",
    "Paraguay": "South America",
    "Peru": "South America",
    "Philippines": "Southeast Asia",
    "Poland": "Eastern Europe",
    "Portugal": "Southern Europe",
    "Puerto Rico": "Caribbean",
    "Qatar": "Western Asia / Middle East",
    "Romania": "Eastern Europe",
    "Russian Federation": "Eastern Europe",
    "Rwanda": "Sub-Saharan Africa",
    "Saudi Arabia": "Western Asia / Middle East",
    "Senegal": "Sub-Saharan Africa",
    "Serbia": "Southern Europe",
    "Sierra Leone": "Sub-Saharan Africa",
    "Singapore": "Southeast Asia",
    "Slovakia": "Eastern Europe",
    "Slovenia": "Southern Europe",
    "Somalia": "Sub-Saharan Africa",
    "Somaliland Region": "Sub-Saharan Africa",
    "Somaliland region": "Sub-Saharan Africa",
    "South Africa": "Sub-Saharan Africa",
    "South Korea": "East Asia",
    "South Sudan": "Sub-Saharan Africa",
    "Spain": "Southern Europe",
    "Sri Lanka": "South Asia",
    "Sudan": "Northern Africa",
    "Suriname": "South America",
    "Swaziland": "Sub-Saharan Africa",
    "Sweden": "Northern Europe",
    "Switzerland": "Western Europe",
    "Syria": "Western Asia / Middle East",
    "Taiwan": "East Asia",
    "Tajikistan": "Central Asia",
    "Tanzania": "Sub-Saharan Africa",
    "Thailand": "Southeast Asia",
    "Togo": "Sub-Saharan Africa",
    "Trinidad and Tobago": "Caribbean",
    "Tunisia": "Northern Africa",
    "Turkey": "Western Asia / Middle East",
    "Turkmenistan": "Central Asia",
    "Uganda": "Sub-Saharan Africa",
    "Ukraine": "Eastern Europe",
    "United Arab Emirates": "Western Asia / Middle East",
    "United Kingdom": "Northern Europe",
    "United States": "North America",
    "Uruguay": "South America",
    "Uzbekistan": "Central Asia",
    "Venezuela": "South America",
    "Vietnam": "Southeast Asia",
    "Yemen": "Western Asia / Middle East",
    "Zambia": "Sub-Saharan Africa",
    "Zimbabwe": "Sub-Saharan Africa",
}


def load_processed_data(filepath: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load pre-processed happiness data from CSV.

    Parameters
    ----------
    filepath : Path
        Path to the processed CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded happiness dataset.

    Raises
    ------
    FileNotFoundError
        If the processed data file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data file not found at {filepath}. Please run process_and_save_data() first."
        )
    logger.info("Loading processed data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def process_and_save_data(
    data_dir: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_PATH,
    validate: bool = True,
) -> pd.DataFrame:
    """Process raw happiness data files and persist the combined dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing raw yearly CSV files.
    output_path : Path
        Path where the combined CSV will be saved.
    validate : bool
        If True, run validation checks on the output data.

    Returns
    -------
    pd.DataFrame
        The combined and processed dataset.

    Raises
    ------
    FileNotFoundError
        If any raw data files are missing.
    ValueError
        If validation fails (when validate=True).
    """
    logger.info("Starting data processing...")

    year_files = {
        2015: data_dir / "2015.csv",
        2016: data_dir / "2016.csv",
        2017: data_dir / "2017.csv",
        2018: data_dir / "2018.csv",
        2019: data_dir / "2019.csv",
    }

    missing = [year for year, path in year_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw data files for years: {missing}")

    logger.info("Loading and combining data from %d years", len(year_files))
    df = load_and_combine_years(
        year_files,
        COLUMN_NORMALIZATION,
        COUNTRY_ALIASES,
        COUNTRY_TO_REGION,
    )

    logger.info("Enriching with geographic metadata")
    df = enrich_with_continents(df, REGION_TO_CONTINENT)

    # Keep only the columns we need, in order
    df = df[[col for col in OUTPUT_COLUMNS if col in df.columns]]

    if validate:
        validate_data(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved combined dataset to %s (%d rows)", output_path, len(df))

    return df


def _normalize_columns(df: pd.DataFrame, year: int, column_normalization: Dict[int, Dict[str, str]]) -> pd.DataFrame:
    mapping = column_normalization.get(year, {})
    df = df.rename(columns=mapping)
    df.columns = [col.strip() for col in df.columns]
    if "country" not in df.columns:
        raise ValueError(f"'country' column missing for {year}")
    return df


def _clean_country_names(df: pd.DataFrame, country_aliases: Dict[str, str]) -> pd.DataFrame:
    df["country"] = df["country"].replace(country_aliases).str.strip()
    return df


def _clean_region_names(df: pd.DataFrame) -> pd.DataFrame:
    if "region" not in df.columns:
        df["region"] = np.nan
    else:
        df["region"] = df["region"].replace({"": np.nan})
        if df["region"].notna().any():
            df["region"] = df["region"].str.strip()
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "happiness_score",
        "gdp_per_capita",
        "social_support",
        "healthy_life_expectancy",
        "freedom",
        "generosity",
        "corruption_perception",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def harmonize_year(
    year: int,
    path: Path,
    column_normalization: Dict[int, Dict[str, str]],
    country_aliases: Dict[str, str],
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df, year, column_normalization)
    df = _clean_country_names(df, country_aliases)
    df = _clean_region_names(df)
    df["year"] = year
    return _convert_numeric_columns(df)


def load_and_combine_years(
    year_files: Dict[int, Path],
    column_normalization: Dict[int, Dict[str, str]],
    country_aliases: Dict[str, str],
    country_to_region: Dict[str, str],
) -> pd.DataFrame:
    frames = []
    for yr, filepath in year_files.items():
        df_year = harmonize_year(yr, filepath, column_normalization, country_aliases)
        frames.append(df_year)

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["region"] = df["country"].map(country_to_region)

    # Fail fast on unmapped countries
    unmapped = df[df["region"].isna()]["country"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Unmapped countries (add to COUNTRY_TO_REGION): {sorted(unmapped)}")

    return df


def enrich_with_continents(
    df: pd.DataFrame,
    region_to_continent: Dict[str, str],
) -> pd.DataFrame:
    """Add continent column based on region.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'region' column already populated.
    region_to_continent : Dict[str, str]
        Mapping from region names to continent names.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'continent' column added.
    """
    df["continent"] = df["region"].map(region_to_continent)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run sanity checks on the processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The processed happiness dataset.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    issues = []

    # Check for required columns
    required_cols = ["country", "year", "happiness_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check for duplicate country-year combinations
    duplicates = df.duplicated(subset=["country", "year"], keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        dup_examples = df[duplicates][["country", "year"]].drop_duplicates().head(5)
        issues.append(f"Found {dup_count} duplicate country-year rows. Examples: {dup_examples.to_dict('records')}")

    # Check row counts per year
    year_counts = df.groupby("year").size()
    logger.info("Row counts per year:\n%s", year_counts.to_string())

    # Warn if any year has unusually few countries
    for year, count in year_counts.items():
        if count < 100:
            logger.warning("Year %d has only %d countries (expected ~150+)", year, count)

    # Check for null happiness scores
    null_scores = df["happiness_score"].isna().sum()
    if null_scores > 0:
        issues.append(f"Found {null_scores} rows with null happiness_score")

    # Check happiness score range (should be roughly 0-10)
    min_score = df["happiness_score"].min()
    max_score = df["happiness_score"].max()
    if min_score < 0 or max_score > 10:
        issues.append(f"Happiness scores outside expected range [0-10]: min={min_score}, max={max_score}")
    logger.info("Happiness score range: %.3f - %.3f", min_score, max_score)

    # Report unique values
    logger.info("Unique countries: %d", df["country"].nunique())
    logger.info("Unique regions: %d", df["region"].nunique())
    logger.info("Unique continents: %d", df["continent"].nunique())
    logger.info("Years covered: %s", sorted(df["year"].unique()))

    if issues:
        for issue in issues:
            logger.error("Validation error: %s", issue)
        raise ValueError(f"Data validation failed with {len(issues)} issue(s)")

    logger.info("Data validation passed ✓")


def main() -> None:
    """Command-line entry point for data processing."""
    parser = argparse.ArgumentParser(
        description="Process World Happiness Report data (2015-2019)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_preprocessor.py                  # Process and save data
  python data_preprocessor.py --validate-only  # Only validate existing data
  python data_preprocessor.py --no-validate    # Skip validation checks
""",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing processed data without regenerating",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation checks during processing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help=f"Output file path (default: {PROCESSED_DATA_PATH})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.validate_only:
        logger.info("Validation-only mode")
        df = load_processed_data()
        validate_data(df)
    else:
        process_and_save_data(
            output_path=args.output,
            validate=not args.no_validate,
        )


if __name__ == "__main__":
    main()
