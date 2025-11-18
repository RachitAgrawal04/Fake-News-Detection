"""Command-line entrypoint for training modern fake news detectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import typer
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score

from .data_utils import (
    DatasetSplits,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    describe_split,
    engineer_features,
    load_splits,
)
from .modeling import ClassifierName, build_pipeline, pretty_classification_report

app = typer.Typer(help="Modern training utilities for the LIAR fake-news dataset.")
console = Console()


def _prepare_splits(data_dir: Path) -> DatasetSplits:
    raw = load_splits(data_dir)
    return DatasetSplits(
        train=engineer_features(raw.train),
        validation=engineer_features(raw.validation),
        test=engineer_features(raw.test),
    )


def _summarize_splits(splits: DatasetSplits) -> None:
    table = Table(title="Label distribution", show_lines=False)
    table.add_column("Split", justify="left")
    table.add_column("label", justify="left")
    table.add_column("share", justify="right")

    for split_name, frame in splits.as_dict().items():
        distribution = describe_split(frame)
        for label, share in distribution.items():
            table.add_row(split_name, str(label), f"{share:.3f}")

    console.print(table)


def _evaluate(model, frame, split_name: str) -> Dict[str, float]:
    X = frame[FEATURE_COLUMNS]
    y_true = frame[LABEL_COLUMN]
    preds = model.predict(X)
    report = pretty_classification_report(y_true, preds)
    console.rule(f"{split_name.title()} classification report")
    console.print(report)
    return {
        "split": split_name,
        "accuracy": accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro"),
        "micro_f1": f1_score(y_true, preds, average="micro"),
    }


@app.command()
def train(
    data_dir: Path = typer.Option(Path("."), help="Folder that stores the TSV splits."),
    classifier: ClassifierName = typer.Option("logreg", help="Classifier to train."),
    max_features: int = typer.Option(40000, help="Max number of TF-IDF features."),
    ngram_max: int = typer.Option(2, help="Highest n-gram to consider."),
    output_dir: Path = typer.Option(Path("artifacts"), help="Where to persist metrics/models."),
    random_state: int = typer.Option(42, help="Random seed for reproducibility."),
    persist_model: bool = typer.Option(True, help="Persist the trained pipeline via joblib."),
) -> None:
    """Train a text + metadata classifier using the LIAR dataset."""

    splits = _prepare_splits(data_dir)
    _summarize_splits(splits)

    model = build_pipeline(
        classifier=classifier,
        max_features=max_features,
        ngram_max=ngram_max,
        random_state=random_state,
    )

    console.rule("Fitting model")
    X_train = splits.train[FEATURE_COLUMNS]
    y_train = splits.train[LABEL_COLUMN]
    model.fit(X_train, y_train)

    metrics = []
    for split_name, frame in splits.as_dict().items():
        metrics.append(_evaluate(model, frame, split_name))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{classifier}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    console.print(f"Saved metrics to {metrics_path.as_posix()}")

    if persist_model:
        model_path = output_dir / f"{classifier}_pipeline.joblib"
        joblib.dump(model, model_path)
        console.print(f"Persisted trained pipeline -> {model_path.as_posix()}")


if __name__ == "__main__":
    app()
