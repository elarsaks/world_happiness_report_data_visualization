"""Self-contained data processing helpers for the World Happiness Report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Project defaults
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_FILENAME = "happiness_combined_2015_2019.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / PROCESSED_FILENAME


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

REGION_OVERRIDES: Dict[str, str] = {
    "Taiwan": "Eastern Asia",
    "Hong Kong": "Eastern Asia",
    "Macedonia": "Central and Eastern Europe",
    "Somaliland Region": "Sub-Saharan Africa",
    "Somalia": "Sub-Saharan Africa",
    "South Sudan": "Sub-Saharan Africa",
    "Gambia": "Sub-Saharan Africa",
    "Lesotho": "Sub-Saharan Africa",
    "Namibia": "Sub-Saharan Africa",
    "Mozambique": "Sub-Saharan Africa",
    "Madagascar": "Sub-Saharan Africa",
    "Laos": "Southeastern Asia",
    "Vietnam": "Southeastern Asia",
    "Myanmar": "Southeastern Asia",
    "Bhutan": "Southern Asia",
    "Bangladesh": "Southern Asia",
    "Cambodia": "Southeastern Asia",
    "Belarus": "Central and Eastern Europe",
    "Serbia": "Central and Eastern Europe",
    "Bosnia and Herzegovina": "Central and Eastern Europe",
    "Albania": "Central and Eastern Europe",
    "Azerbaijan": "Central and Eastern Europe",
    "Kazakhstan": "Central and Eastern Europe",
    "Kyrgyzstan": "Central and Eastern Europe",
    "Mongolia": "Eastern Asia",
    "Armenia": "Central and Eastern Europe",
    "Georgia": "Central and Eastern Europe",
    "Uzbekistan": "Central and Eastern Europe",
    "Turkmenistan": "Central and Eastern Europe",
    "Tajikistan": "Central and Eastern Europe",
    "Ukraine": "Central and Eastern Europe",
    "Palestinian Territories": "Middle East and Northern Africa",
    "Bahrain": "Middle East and Northern Africa",
    "Qatar": "Middle East and Northern Africa",
    "Saudi Arabia": "Middle East and Northern Africa",
    "United Arab Emirates": "Middle East and Northern Africa",
    "Oman": "Middle East and Northern Africa",
    "Kuwait": "Middle East and Northern Africa",
    "Yemen": "Middle East and Northern Africa",
    "Iraq": "Middle East and Northern Africa",
    "Iran": "Middle East and Northern Africa",
    "Lebanon": "Middle East and Northern Africa",
    "Jordan": "Middle East and Northern Africa",
    "Algeria": "Middle East and Northern Africa",
    "Morocco": "Middle East and Northern Africa",
    "Tunisia": "Middle East and Northern Africa",
}

CONTINENT_OVERRIDES: Dict[str, str] = {
    "Hong Kong": "Asia",
    "Taiwan": "Asia",
    "Somaliland Region": "Africa",
    "Palestinian Territories": "Asia",
    "Kosovo": "Europe",
    "Macedonia": "Europe",
    "Congo (Brazzaville)": "Africa",
    "Congo (Kinshasa)": "Africa",
    "Trinidad and Tobago": "Americas",
    "United Arab Emirates": "Asia",
    "Saudi Arabia": "Asia",
    "Qatar": "Asia",
    "Bahrain": "Asia",
    "Oman": "Asia",
    "Kuwait": "Asia",
    "Yemen": "Asia",
    "Iraq": "Asia",
    "Iran": "Asia",
}

REGION_TO_CONTINENT: Dict[str, str] = {
    "Western Europe": "Europe",
    "Central and Eastern Europe": "Europe",
    "Eastern Asia": "Asia",
    "Southeastern Asia": "Asia",
    "Southern Asia": "Asia",
    "Central Asia": "Asia",
    "Australia and New Zealand": "Oceania",
    "North America": "Americas",
    "Latin America and Caribbean": "Americas",
    "Sub-Saharan Africa": "Africa",
    "Middle East and Northern Africa": "Middle East & North Africa",
    "Commonwealth of Independent States": "Europe & Central Asia",
}


def load_processed_data(filepath: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load pre-processed happiness data from CSV."""

    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data file not found at {filepath}. Please run process_and_save_data() first."
        )
    return pd.read_csv(filepath)


def process_and_save_data(
    data_dir: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Process raw happiness data files and persist the combined dataset."""

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

    df = load_and_combine_years(year_files, COLUMN_NORMALIZATION, COUNTRY_ALIASES, REGION_OVERRIDES)
    df = enrich_with_continents(df, REGION_TO_CONTINENT, CONTINENT_OVERRIDES)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")

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
        "happiness_rank",
        "happiness_score",
        "gdp_per_capita",
        "social_support",
        "healthy_life_expectancy",
        "freedom",
        "generosity",
        "corruption_perception",
        "dystopia_residual",
        "standard_error",
        "whisker_low",
        "whisker_high",
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
    region_overrides: Dict[str, str],
) -> pd.DataFrame:
    frames = []
    region_lookup: Dict[str, str] = {}

    for yr, filepath in year_files.items():
        df_year = harmonize_year(yr, filepath, column_normalization, country_aliases)
        frames.append(df_year)
        if df_year["region"].notna().any():
            year_regions = df_year.dropna(subset=["region"])[["country", "region"]]
            region_lookup.update(year_regions.set_index("country")["region"].to_dict())

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["region"] = df["region"].fillna(df["country"].map(region_lookup))
    df["region"] = df["region"].fillna(df["country"].map(region_overrides))

    return df


def enrich_with_continents(
    df: pd.DataFrame,
    region_to_continent: Dict[str, str],
    continent_overrides: Dict[str, str],
) -> pd.DataFrame:
    # Delay import to keep dependencies light for users who only need CSV loading.
    import plotly.express as px

    gapminder_continents = px.data.gapminder()[["country", "continent"]].drop_duplicates()
    continent_lookup = gapminder_continents.set_index("country")["continent"].to_dict()

    df["continent"] = df["region"].map(region_to_continent)
    df["continent"] = df["continent"].fillna(df["country"].map(continent_lookup))
    df["continent"] = df["continent"].fillna(df["country"].map(continent_overrides))
    df["region"] = df["region"].fillna(df["continent"])
    df["region"] = df["region"].fillna("Unknown")

    return df
