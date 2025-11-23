"""Data normalization functions for World Happiness Report data."""

from pathlib import Path

import numpy as np
import pandas as pd


def harmonize_year(year: int, path: Path, column_normalization: dict, country_aliases: dict) -> pd.DataFrame:
    """Load and standardize a single World Happiness Report file."""
    df = pd.read_csv(path)
    df = _normalize_columns(df, year, column_normalization)
    df = _clean_country_names(df, country_aliases)
    df = _clean_region_names(df)
    df["year"] = year
    df = _convert_numeric_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame, year: int, column_normalization: dict) -> pd.DataFrame:
    """Rename and strip whitespace from column names."""
    df = df.rename(columns=column_normalization.get(year, {}))
    df.columns = [col.strip() for col in df.columns]
    if "country" not in df.columns:
        raise ValueError(f"'country' column missing for {year}")
    return df


def _clean_country_names(df: pd.DataFrame, country_aliases: dict) -> pd.DataFrame:
    """Apply country aliases and strip whitespace."""
    df["country"] = df["country"].replace(country_aliases).str.strip()
    return df


def _clean_region_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean region column, handling missing values."""
    if "region" not in df.columns:
        df["region"] = np.nan
    else:
        df["region"] = df["region"].replace({"": np.nan})
        if df["region"].notna().any():
            df["region"] = df["region"].str.strip()
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to numeric dtype."""
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


def load_and_combine_years(
    year_files: dict,
    column_normalization: dict,
    country_aliases: dict,
    region_overrides: dict,
) -> pd.DataFrame:
    """Load all yearly data and combine into a single DataFrame."""
    frames = []
    region_lookup = {}

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
    region_to_continent: dict,
    continent_overrides: dict,
) -> pd.DataFrame:
    """Add continent information using multiple sources."""
    import plotly.express as px

    # Get continent lookup from Plotly's gapminder
    gapminder_continents = px.data.gapminder()[["country", "continent"]].drop_duplicates()
    continent_lookup = gapminder_continents.set_index("country")["continent"].to_dict()

    # Map continents from multiple sources
    df["continent"] = df["region"].map(region_to_continent)
    df["continent"] = df["continent"].fillna(df["country"].map(continent_lookup))
    df["continent"] = df["continent"].fillna(df["country"].map(continent_overrides))

    # Use continent as fallback for missing regions
    df["region"] = df["region"].fillna(df["continent"])
    df["region"] = df["region"].fillna("Unknown")

    return df
