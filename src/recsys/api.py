"""Recommendation API over the saved artifacts."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from scipy import sparse

from recsys.models.base import top_k, zscore
from recsys.models.classic import ImplicitALS

ModelName = Literal["hybrid", "als", "item_knn", "popularity"]


class Movie(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float


class Recommendations(BaseModel):
    model: str
    items: list[Movie]


class NewUser(BaseModel):
    liked_movie_ids: list[int] = Field(min_length=1, max_length=200)
    k: int = Field(default=10, ge=1, le=50)


class Store:
    def __init__(self, directory: Path) -> None:
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(f"No artifacts in {directory}; run `recsys train` first")
        meta = json.loads((directory / "meta.json").read_text())
        factors = np.load(directory / "als.npz")
        self.als = ImplicitALS(
            alpha=meta["als"]["alpha"], regularization=meta["als"]["regularization"]
        )
        self.als.user_factors, self.als.item_factors = (
            factors["user_factors"],
            factors["item_factors"],
        )
        self.als.factors = self.als.item_factors.shape[1]
        self.knn = sparse.load_npz(directory / "item_knn.npz").tocsr()
        self.train = sparse.load_npz(directory / "train.npz").tocsr()
        self.content = np.load(directory / "content.npy")
        self.movies = pd.read_csv(directory / "movies.csv")
        self.users = {u: i for i, u in enumerate(meta["users"])}
        self.items = {int(m): i for i, m in enumerate(self.movies["movieId"])}
        self.popularity = np.asarray(self.train.sum(axis=0)).ravel().astype(np.float32)

    def scores(self, model: str, history: list[int], user: int | None) -> np.ndarray:
        if model == "popularity":
            return self.popularity
        if model == "item_knn":
            vec = np.zeros(len(self.items), dtype=np.float32)
            vec[history] = 1.0
            return np.asarray(self.knn.T @ vec).ravel()
        als = (
            self.als.item_factors @ self.als.user_factors[user]
            if user is not None
            else self.als.fold_in(history)
        )
        if model == "als":
            return als.astype(np.float32)
        profile = self.content[history].mean(axis=0) if history else np.zeros(self.content.shape[1])
        w = 1.0 / (1.0 + len(history) / 10.0)
        return (1 - w) * zscore(als.astype(np.float32)) + w * zscore(self.content @ profile)

    def recommend(
        self, model: str, history: list[int], k: int, user: int | None = None
    ) -> list[Movie]:
        scores = self.scores(model, history, user)
        picks = top_k(scores.astype(np.float32), np.array(history, dtype=np.int32), k)
        return [self._movie(i, float(scores[i])) for i in picks]

    def _movie(self, i: int, score: float) -> Movie:
        row = self.movies.iloc[int(i)]
        return Movie(
            movie_id=int(row.movieId), title=row.title, genres=row.genres, score=round(score, 4)
        )


@lru_cache
def get_store(directory: str) -> Store:
    return Store(Path(directory))


def create_app(directory: Path | None = None) -> FastAPI:
    path = str(directory) if directory else os.getenv("RECSYS_ARTIFACTS", "artifacts")
    app = FastAPI(title="recommender-system", version="0.1.0")

    def store() -> Store:
        try:
            return get_store(path)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, int | str]:
        s = store()
        return {"status": "ok", "users": len(s.users), "items": len(s.items)}

    @app.get("/users/{user_id}/recommendations")
    def for_user(
        user_id: int, k: int = Query(10, ge=1, le=50), model: ModelName = "hybrid"
    ) -> Recommendations:
        s = store()
        if user_id not in s.users:
            raise HTTPException(404, "Unknown user; use POST /recommendations for new users")
        user = s.users[user_id]
        history = s.train[user].indices.tolist()
        return Recommendations(model=model, items=s.recommend(model, history, k, user))

    @app.post("/recommendations")
    def for_new_user(body: NewUser, model: ModelName = "hybrid") -> Recommendations:
        s = store()
        history = [s.items[m] for m in body.liked_movie_ids if m in s.items]
        if not history:
            raise HTTPException(422, "None of the movie ids are in the catalogue")
        return Recommendations(model=model, items=s.recommend(model, history, body.k))

    @app.get("/movies/{movie_id}/similar")
    def similar(movie_id: int, k: int = Query(10, ge=1, le=50)) -> Recommendations:
        s = store()
        if movie_id not in s.items:
            raise HTTPException(404, "Unknown movie")
        item = s.items[movie_id]
        scores = np.asarray(s.knn[item].todense()).ravel()
        picks = top_k(scores.astype(np.float32), np.array([item], dtype=np.int32), k)
        return Recommendations(
            model="item_knn", items=[s._movie(i, float(scores[i])) for i in picks]
        )

    return app
