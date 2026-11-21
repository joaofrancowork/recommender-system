"""MovieLens loading, implicit-feedback conversion and a temporal per-user split.

Dataset: MovieLens latest-small (GroupLens Research, University of Minnesota).
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM TiiS 5, 4. https://doi.org/10.1145/2827872
The licence does not allow redistribution, so the data is downloaded at run time.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

log = logging.getLogger(__name__)
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data/ml-latest-small")
POSITIVE_RATING = 3.5


def download(data_dir: Path = DATA_DIR, url: str = URL) -> Path:
    if (data_dir / "ratings.csv").exists():
        return data_dir
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ratings.csv", "movies.csv", "tags.csv"):
        (data_dir / name).write_bytes(archive.read(f"ml-latest-small/{name}"))
    return data_dir


def load(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    tags_path = data_dir / "tags.csv"
    if tags_path.exists():
        tags = (
            pd.read_csv(tags_path)
            .groupby("movieId")["tag"]
            .agg(lambda t: " ".join(sorted(set(map(str, t)))))
        )
        movies = movies.merge(tags.rename("tags"), on="movieId", how="left")
    else:
        movies["tags"] = ""
    movies["tags"] = movies["tags"].fillna("")
    return ratings, movies


def to_implicit(ratings: pd.DataFrame, threshold: float = POSITIVE_RATING) -> pd.DataFrame:
    """Keep only ratings that signal a positive experience: 'would recommend'."""
    return ratings[ratings["rating"] >= threshold].reset_index(drop=True)


def temporal_split(
    interactions: pd.DataFrame, test_frac: float = 0.2, min_interactions: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent interactions.

    Predicting what a user watches *next* from what they watched before is the real
    task; a random split lets the model see the future and inflates every metric.
    """
    ordered = interactions.sort_values(["userId", "timestamp", "movieId"])
    position = ordered.groupby("userId").cumcount()
    size = ordered.groupby("userId")["movieId"].transform("size")
    n_test = np.floor(size * test_frac).astype(int)
    is_test = (size >= min_interactions) & (position >= size - n_test)
    return ordered[~is_test].reset_index(drop=True), ordered[is_test].reset_index(drop=True)


@dataclass
class Index:
    """Contiguous integer ids for users and items, from the training data."""

    users: dict[int, int]
    items: dict[int, int]
    item_ids: np.ndarray

    @classmethod
    def build(cls, train: pd.DataFrame, movies: pd.DataFrame) -> Index:
        users = {u: i for i, u in enumerate(sorted(train["userId"].unique()))}
        item_ids = np.array(sorted(movies["movieId"].unique()))
        return cls(users, {m: i for i, m in enumerate(item_ids)}, item_ids)

    def matrix(self, interactions: pd.DataFrame) -> sparse.csr_matrix:
        known = interactions[
            interactions["userId"].isin(self.users.keys())
            & interactions["movieId"].isin(self.items.keys())
        ].drop_duplicates(["userId", "movieId"])  # binary feedback: repeats would add up
        rows = known["userId"].map(self.users).to_numpy()
        cols = known["movieId"].map(self.items).to_numpy()
        data = np.ones(len(known), dtype=np.float32)
        shape = (len(self.users), len(self.items))
        return sparse.csr_matrix((data, (rows, cols)), shape=shape)
