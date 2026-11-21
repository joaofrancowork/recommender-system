"""Ranking metrics on the held-out future of each user.

* Recall@k and NDCG@k: did we surface what the user actually watched next, and how high?
* Coverage: share of the catalogue that ever gets recommended (popularity scores ~0).
* Novelty: mean self-information of recommended items, -log2(popularity); higher
  means less obvious recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.data import Index
from recsys.models.base import Recommender, top_k


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    gains = [1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(sum(gains) / ideal) if ideal else 0.0


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / min(len(relevant), k)


@dataclass
class Report:
    model: str
    users: int
    recall: float
    ndcg: float
    coverage: float
    novelty: float
    cold_recall: float
    cold_users: int

    def as_row(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def evaluate(
    model: Recommender,
    train: sparse.csr_matrix,
    test: pd.DataFrame,
    index: Index,
    k: int = 10,
    cold_threshold: int = 15,
) -> Report:
    popularity = np.asarray(train.sum(axis=0)).ravel()
    p_item = (popularity + 1) / (popularity.sum() + len(popularity))
    known = test[test["userId"].isin(index.users) & test["movieId"].isin(index.items)]
    relevant_by_user: dict[int, set[int]] = {}
    for user_id, movie_id in zip(
        known["userId"].astype(int), known["movieId"].astype(int), strict=True
    ):
        relevant_by_user.setdefault(user_id, set()).add(index.items[movie_id])
    recalls, ndcgs, novelty, cold = [], [], [], []
    recommended_items: set[int] = set()
    for user_id, relevant in relevant_by_user.items():
        user = index.users[user_id]
        seen = train[user].indices
        recs = top_k(model.score(user, train), seen, k).tolist()
        recommended_items.update(recs)
        r = recall_at_k(recs, relevant, k)
        recalls.append(r)
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        novelty.extend(-np.log2(p_item[recs]))
        if len(seen) <= cold_threshold:
            cold.append(r)
    return Report(
        model=model.name,
        users=len(recalls),
        recall=float(np.mean(recalls)) if recalls else 0.0,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        coverage=len(recommended_items) / train.shape[1],
        novelty=float(np.mean(novelty)) if novelty else 0.0,
        cold_recall=float(np.mean(cold)) if cold else 0.0,
        cold_users=len(cold),
    )


def to_markdown(reports: list[Report], k: int = 10) -> str:
    lines = [
        f"| Model | Recall@{k} | NDCG@{k} | Coverage | Novelty (bits) "
        f"| Recall@{k}, short histories |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.model} | {r.recall:.3f} | {r.ndcg:.3f} | {r.coverage * 100:.1f}% | "
            f"{r.novelty:.1f} | {r.cold_recall:.3f} ({r.cold_users} users) |"
        )
    return "\n".join(lines)
