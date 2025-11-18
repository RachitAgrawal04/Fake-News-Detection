"""Utilities for loading and featurising the LIAR fake-news detection dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

LIAR_COLUMNS = [
    "row_id",
    "statement_id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job",
    "state_info",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
    "justification",
]

TEXT_COLUMNS = ["statement", "subjects", "context", "justification"]
CATEGORICAL_COLUMNS = ["speaker", "party", "state_info", "speaker_job"]
NUMERIC_COLUMNS = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
]
LABEL_COLUMN = "label"
FEATURE_COLUMNS = ["text", *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]


@dataclass(slots=True)
class DatasetSplits:
    """Container for the LIAR dataset splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="warn",
        encoding="utf-8",
    )
    return df


def load_splits(data_dir: Path) -> DatasetSplits:
    """Load train/validation/test TSV files into memory."""

    train = _load_split(data_dir / "train_data.tsv")
    validation = _load_split(data_dir / "Validation_data.tsv")
    test = _load_split(data_dir / "test_data.tsv")
    return DatasetSplits(train=train, validation=validation, test=test)


def _clean_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df:
            continue
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.replace("\u2019", "'", regex=False)
            .str.strip()
        )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with helper columns for modeling."""

    processed = df.copy(deep=False)
    processed = _clean_text_columns(processed, TEXT_COLUMNS)

    processed["text"] = (
        processed[TEXT_COLUMNS]
        .fillna("")
        .agg(lambda row: " ".join(filter(None, row)), axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    processed.loc[processed["text"].eq(""), "text"] = processed.loc[
        processed["text"].eq(""), "statement"
    ]

    for cat_col in CATEGORICAL_COLUMNS:
        processed[cat_col] = processed[cat_col].fillna("unknown").astype(str).str.lower()

    processed[NUMERIC_COLUMNS] = (
        processed[NUMERIC_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype("float64")
    )

    processed[LABEL_COLUMN] = processed[LABEL_COLUMN].astype("category")
    return processed


def describe_split(df: pd.DataFrame) -> pd.Series:
    """Return a simple label distribution for quick sanity checks."""

    return df[LABEL_COLUMN].value_counts(normalize=True).sort_index()
