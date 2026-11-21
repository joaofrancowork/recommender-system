"""Sentence embeddings for movies (title, genres and user tags) with fastembed (local ONNX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


def movie_text(movies: pd.DataFrame) -> list[str]:
    return [
        f"{title}. Genres: {genres.replace('|', ', ')}. Tags: {tags}".strip()
        for title, genres, tags in zip(
            movies["title"].astype(str),
            movies["genres"].astype(str),
            movies["tags"].astype(str),
            strict=True,
        )
    ]


def embed_movies(
    movies: pd.DataFrame, cache: Path, model: str = "BAAI/bge-small-en-v1.5"
) -> npt.NDArray[np.float32]:
    if cache.exists():
        return np.load(cache)  # type: ignore[no-any-return]
    from fastembed import TextEmbedding

    vectors = np.asarray(list(TextEmbedding(model_name=model).embed(movie_text(movies))))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors.astype(np.float32))
    return vectors.astype(np.float32)
