from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recsys import cli
from recsys.api import create_app, get_store
from recsys.evaluation import evaluate, to_markdown
from recsys.models.classic import ImplicitALS, ItemKNN, Popularity
from recsys.models.content import ContentBased, Hybrid, genre_features
from recsys.pipeline import Dataset, build_models, evaluate_all, prepare, save_artifacts, tune


@pytest.fixture
def ds(movielens: Path) -> Dataset:
    return prepare(movielens)


def test_collaborative_models_beat_popularity_on_taste_clusters(ds: Dataset) -> None:
    pop = evaluate(Popularity().fit(ds.train), ds.train, ds.test, ds.index)
    for model in (ItemKNN(neighbours=20), ImplicitALS(factors=8, iterations=8)):
        report = evaluate(model.fit(ds.train), ds.train, ds.test, ds.index)
        assert report.recall > pop.recall, model.name
        assert report.coverage > pop.coverage


def test_content_and_hybrid(ds: Dataset) -> None:
    features = genre_features(ds.movies["genres"].tolist())
    assert features.shape[1] == 5 and features.sum(axis=1).max() == 1
    content = ContentBased(features).fit(ds.train)
    hybrid = Hybrid(ImplicitALS(factors=8, iterations=5), content).fit(ds.train)
    assert hybrid.content_weight(0) == 1.0 and hybrid.content_weight(10) == 0.5
    assert evaluate(hybrid, ds.train, ds.test, ds.index).recall > 0.2


def test_content_handles_empty_history(ds: Dataset) -> None:
    content = ContentBased(genre_features(ds.movies["genres"].tolist()))
    empty = ds.train.copy()
    empty.data[:] = 0
    empty.eliminate_zeros()
    assert np.all(content.score(0, empty) == 0)


def test_als_fold_in_matches_taste(ds: Dataset) -> None:
    als = ImplicitALS(factors=8, iterations=8).fit(ds.train)
    action = [i for i, g in enumerate(ds.movies["genres"]) if g == "Action"][:3]
    picks = np.argsort(-als.fold_in(action))[:10]
    assert sum(ds.movies["genres"].iloc[i] == "Action" for i in picks) >= 7
    assert not als.fold_in([]).any()


def test_tune_picks_from_grid(ds: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    import recsys.pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "ALS_GRID", {"factors": [4, 8], "regularization": [0.1], "alpha": [10.0]}
    )
    monkeypatch.setattr(pipeline, "KNN_GRID", {"neighbours": [5, 20]})
    best = tune(ds)
    assert best["als"]["factors"] in {4, 8} and best["item_knn"]["neighbours"] in {5, 20}


@pytest.fixture
def artifacts(ds: Dataset, tmp_path: Path) -> Path:
    params = {"als": {"factors": 8, "iterations": 8}, "item_knn": {"neighbours": 20}}
    models = build_models(ds, None, params)
    table = to_markdown(evaluate_all(ds, models))
    assert "| hybrid |" in table
    out = tmp_path / "artifacts"
    save_artifacts(ds, models, out, params)
    get_store.cache_clear()
    return out


def test_api_endpoints(artifacts: Path, ds: Dataset) -> None:
    client = TestClient(create_app(artifacts))
    assert client.get("/health").json()["users"] == len(ds.index.users)
    user_id = next(iter(ds.index.users))
    for model in ("hybrid", "als", "item_knn", "popularity"):
        res = client.get(f"/users/{user_id}/recommendations", params={"k": 5, "model": model})
        assert res.status_code == 200 and len(res.json()["items"]) == 5, model
    assert client.get("/users/99999/recommendations").status_code == 404

    new = client.post("/recommendations", json={"liked_movie_ids": [1, 6, 11], "k": 5}).json()
    liked_titles = {"Movie 1 (1991)", "Movie 6 (1996)", "Movie 11 (2001)"}
    reasons = [m["because_you_liked"] for m in new["items"]]
    assert any(reasons) and all(r in liked_titles for r in reasons if r)
    assert len(new["items"]) == 5 and all(m["movie_id"] not in {1, 6, 11} for m in new["items"])
    assert client.post("/recommendations", json={"liked_movie_ids": [123456]}).status_code == 422

    similar = client.get("/movies/1/similar", params={"k": 3}).json()["items"]
    assert len(similar) == 3 and all(m["movie_id"] != 1 for m in similar)
    assert client.get("/movies/123456/similar").status_code == 404


def test_explanation_is_none_without_a_link(artifacts: Path) -> None:
    store = get_store(str(artifacts))
    assert store.explain(0, []) is None
    unrelated = np.where(np.asarray(store.knn[0].todense()).ravel() == 0)[0]
    assert store.explain(int(unrelated[1]), [0]) is None


def test_api_without_artifacts(tmp_path: Path) -> None:
    get_store.cache_clear()
    assert TestClient(create_app(tmp_path)).get("/health").status_code == 503


def test_cli_train_with_genre_content(
    movielens: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "a"
    assert (
        cli.main(["train", "--data", str(movielens), "--out", str(out), "--content", "genres"]) == 0
    )
    assert "| als |" in capsys.readouterr().out
    assert (out / "meta.json").exists() and (out / "evaluation.md").exists()
