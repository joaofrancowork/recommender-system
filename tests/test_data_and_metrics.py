from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from recsys.data import Index, download, load, temporal_split, to_implicit
from recsys.evaluation import ndcg_at_k, recall_at_k
from recsys.models.base import top_k, zscore


def test_download_extracts_and_caches(tmp_path: Path, movielens: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("ratings.csv", "movies.csv", "tags.csv"):
            z.write(movielens / name, f"ml-latest-small/{name}")
    target = tmp_path / "dl"
    with patch("urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = io.BytesIO(archive.getvalue())
        download(target)
        download(target)
    assert urlopen.call_count == 1 and (target / "ratings.csv").exists()


def test_load_merges_tags_and_implicit_filter(movielens: Path) -> None:
    ratings, movies = load(movielens)
    assert movies.loc[movies.movieId == 1, "tags"].item() == "classic"
    assert movies.loc[movies.movieId == 2, "tags"].item() == ""
    assert (to_implicit(ratings)["rating"] >= 3.5).all()
    (movielens / "tags.csv").unlink()
    assert (load(movielens)[1]["tags"] == "").all()


def test_temporal_split_holds_out_each_users_latest() -> None:
    df = pd.DataFrame(
        {
            "userId": [1] * 10 + [2] * 3,
            "movieId": range(13),
            "timestamp": [*list(range(10)), 5, 6, 7],
        }
    )
    train, test = temporal_split(df, test_frac=0.2, min_interactions=5)
    assert sorted(test["movieId"]) == [8, 9]  # user 1's two most recent; user 2 too short
    assert train.groupby("userId")["timestamp"].max()[1] < test["timestamp"].min()


def test_index_matrix_ignores_unknown_ids(movielens: Path) -> None:
    ratings, movies = load(movielens)
    index = Index.build(ratings, movies)
    extra = pd.DataFrame({"userId": [999], "movieId": [1]})
    matrix = index.matrix(pd.concat([ratings, extra]))
    unique_pairs = len(ratings.drop_duplicates(["userId", "movieId"]))
    assert matrix.shape == (len(index.users), len(movies)) and matrix.nnz == unique_pairs
    assert matrix.max() == 1.0


def test_ranking_helpers() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    assert top_k(scores, np.array([1], dtype=np.int32), 2).tolist() == [3, 2]
    assert top_k(scores, np.arange(4, dtype=np.int32), 2).tolist() == []
    assert np.allclose(zscore(np.ones(3, dtype=np.float32)), 0)
    assert recall_at_k([1, 2, 3], {3, 9}, 3) == 0.5 and recall_at_k([1], set(), 3) == 0.0
    assert ndcg_at_k([9, 3], {9, 3}, 2) == 1.0
    assert 0 < ndcg_at_k([1, 9], {9}, 2) < 1 and ndcg_at_k([1], set(), 2) == 0.0
