"""Model factories and pipelines for fake-news detection."""

from __future__ import annotations

from typing import Literal, Optional

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC

from .data_utils import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

ClassifierName = Literal["logreg", "linear_svc", "sgd"]


def _build_estimator(name: ClassifierName, random_state: Optional[int] = None):
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            solver="saga",
            penalty="l2",
            n_jobs=-1,
            verbose=0,
            multi_class="multinomial",
            random_state=random_state,
        )
    if name == "linear_svc":
        return LinearSVC(C=1.0, random_state=random_state)
    if name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier '{name}'")


def build_pipeline(
    classifier: ClassifierName = "logreg",
    max_features: int = 40000,
    ngram_max: int = 2,
    random_state: Optional[int] = None,
) -> Pipeline:
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MaxAbsScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "text"),
            ("categorical", categorical_encoder, CATEGORICAL_COLUMNS),
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
        ],
        sparse_threshold=0.3,
    )

    model = _build_estimator(classifier, random_state=random_state)

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def pretty_classification_report(y_true, y_pred) -> str:
    return str(classification_report(y_true=y_true, y_pred=y_pred, digits=3))


__all__ = [
    "build_pipeline",
    "pretty_classification_report",
    "ClassifierName",
    "FEATURE_COLUMNS",
]
