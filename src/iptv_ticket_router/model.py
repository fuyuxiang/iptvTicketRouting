from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import jieba
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .text_preprocess import normalize_text


def jieba_tokenize(text: str) -> List[str]:
    return jieba.lcut(text, cut_all=False)


@dataclass
class ModelMeta:
    created_at_unix: int
    sklearn_version: str
    classes: List[str]
    pipeline: str = "tfidf + logistic_regression"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at_unix": self.created_at_unix,
            "sklearn_version": self.sklearn_version,
            "classes": self.classes,
            "pipeline": self.pipeline,
        }


class TicketClassifier:
    """Text classifier for ticket fault category (故障大类)."""

    def __init__(self) -> None:
        self.pipeline: Pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    tokenizer=jieba_tokenize,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    C=2.0,
                    class_weight="balanced",
                ),
            ),
        ])

    def fit(self, texts: Sequence[str], y: Sequence[str]) -> "TicketClassifier":
        self.pipeline.fit(texts, y)
        return self

    def predict(self, texts: Sequence[str]) -> List[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: Sequence[str]):
        return self.pipeline.predict_proba(texts)

    def classes_(self) -> List[str]:
        clf = self.pipeline.named_steps["clf"]
        return list(getattr(clf, "classes_", []))

    def save(self, model_path: str, label_map_path: str, meta_path: str | None = None) -> None:
        Path(os.path.dirname(model_path) or ".").mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_path)

        label_map = {"classes_": self.classes_()}
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

        if meta_path:
            meta = ModelMeta(
                created_at_unix=int(time.time()),
                sklearn_version=sklearn.__version__,
                classes=self.classes_(),
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, model_path: str) -> "TicketClassifier":
        self.pipeline = joblib.load(model_path)
        return self

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
