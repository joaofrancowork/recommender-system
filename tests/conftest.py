from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance"]


def write_movielens(directory: Path, users: int = 60, movies: int = 80, seed: int = 3) -> Path:
    """Tiny MovieLens-shaped dataset with taste clusters: each user mostly likes one genre."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    movie_genre = [GENRES[i % len(GENRES)] for i in range(movies)]
    pd.DataFrame(
        {
            "movieId": np.arange(1, movies + 1),
            "title": [f"Movie {i} ({1990 + i % 30})" for i in range(1, movies + 1)],
            "genres": movie_genre,
        }
    ).to_csv(directory / "movies.csv", index=False)
    rows = []
    for u in range(1, users + 1):
        taste = GENRES[u % len(GENRES)]
        liked = [m for m in range(1, movies + 1) if movie_genre[m - 1] == taste]
        picks = rng.choice(liked, size=10, replace=False)
        noise = rng.choice(np.arange(1, movies + 1), size=3, replace=False)
        for t, m in enumerate([*picks, *noise]):
            rating = 4.5 if m in picks else 2.0
            rows.append((u, int(m), rating, 1_000_000 + u * 1000 + t))
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(
        directory / "ratings.csv", index=False
    )
    pd.DataFrame({"userId": [1], "movieId": [1], "tag": ["classic"], "timestamp": [1]}).to_csv(
        directory / "tags.csv", index=False
    )
    return directory


@pytest.fixture
def movielens(tmp_path: Path) -> Path:
    return write_movielens(tmp_path / "ml")
