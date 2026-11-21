"""Build data, tune on validation, fit the final models and evaluate on test."""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.data import Index, load, temporal_split, to_implicit
from recsys.evaluation import Report, evaluate
from recsys.models.base import Recommender
from recsys.models.classic import ImplicitALS, ItemKNN, Popularity
from recsys.models.content import ContentBased, Hybrid, genre_features

log = logging.getLogger(__name__)

ALS_GRID: dict[str, list[Any]] = {
    "factors": [32, 64, 128],
    "regularization": [0.01, 0.1],
    "alpha": [10.0, 40.0],
}
KNN_GRID: dict[str, list[Any]] = {"neighbours": [50, 100, 200]}


@dataclass
class Dataset:
    movies: pd.DataFrame
    index: Index
    train: sparse.csr_matrix
    test: pd.DataFrame
    train_df: pd.DataFrame


def prepare(data_dir: Path) -> Dataset:
    ratings, movies = load(data_dir)
    train_df, test = temporal_split(to_implicit(ratings))
    index = Index.build(train_df, movies)
    movies = movies.set_index("movieId").loc[index.item_ids].reset_index()
    return Dataset(movies, index, index.matrix(train_df), test, train_df)


def _grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]


def tune(ds: Dataset) -> dict[str, dict[str, Any]]:
    """Pick hyper-parameters on a validation split carved from the *training* period."""
    inner_train, validation = temporal_split(ds.train_df)
    inner = ds.index.matrix(inner_train)
    best: dict[str, dict[str, Any]] = {}
    searches: list[tuple[str, dict[str, list[Any]], Callable[[dict[str, Any]], Recommender]]] = [
        ("als", ALS_GRID, lambda p: ImplicitALS(**p, iterations=10)),
        ("item_knn", KNN_GRID, lambda p: ItemKNN(**p)),
    ]
    for name, grid, make in searches:
        scored = []
        for params in _grid(grid):
            report = evaluate(make(params).fit(inner), inner, validation, ds.index)
            scored.append((report.ndcg, params))
            log.info("%s %s val NDCG@10 %.4f", name, params, report.ndcg)
        best[name] = max(scored, key=lambda t: t[0])[1]
    return best


def build_models(
    ds: Dataset,
    embeddings: np.ndarray | None,
    params: dict[str, dict[str, Any]],
) -> list[Recommender]:
    als_params = params.get("als", {})
    content_vectors = (
        embeddings if embeddings is not None else genre_features(ds.movies["genres"].tolist())
    )
    models: list[Recommender] = [
        Popularity(),
        ItemKNN(**params.get("item_knn", {})),
        ImplicitALS(**als_params),
        ContentBased(content_vectors),
        Hybrid(ImplicitALS(**als_params), ContentBased(content_vectors)),
    ]
    return [m.fit(ds.train) for m in models]


def evaluate_all(ds: Dataset, models: list[Recommender]) -> list[Report]:
    return [evaluate(m, ds.train, ds.test, ds.index) for m in models]


def save_artifacts(
    ds: Dataset, models: list[Recommender], out: Path, params: dict[str, Any]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_name = {m.name: m for m in models}
    als = by_name["als"]
    knn = by_name["item_knn"]
    content = by_name["content"]
    assert isinstance(als, ImplicitALS) and isinstance(knn, ItemKNN)
    assert isinstance(content, ContentBased)
    np.savez_compressed(
        out / "als.npz", user_factors=als.user_factors, item_factors=als.item_factors
    )
    sparse.save_npz(out / "item_knn.npz", knn.similarity)
    sparse.save_npz(out / "train.npz", ds.train)
    np.save(out / "content.npy", content.embeddings)
    ds.movies[["movieId", "title", "genres"]].to_csv(out / "movies.csv", index=False)
    meta = {
        "users": [int(u) for u in ds.index.users],
        "params": params,
        "als": {"alpha": als.alpha, "regularization": als.regularization},
    }
    (out / "meta.json").write_text(json.dumps(meta))
