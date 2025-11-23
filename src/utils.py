"""Utility functions for data processing."""

from pathlib import Path

import pandas as pd


def load_simple_mapping(csv_path: Path, key_col: str, value_col: str) -> dict:
    """Load a simple two-column mapping from CSV."""
    df = pd.read_csv(csv_path)
    return dict(zip(df[key_col], df[value_col]))


def load_column_normalization(csv_path: Path) -> dict:
    """Load column normalization mappings from CSV, grouped by year."""
    df = pd.read_csv(csv_path)
    return (
        df.groupby("year")
        .apply(
            lambda g: dict(zip(g["original"], g["normalized"])),
            include_groups=False,  # type: ignore
        )
        .to_dict()
    )  # type: ignore
