from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy import sparse

Scores = npt.NDArray[np.float32]


class Recommender(Protocol):
    name: str

    def fit(self, interactions: sparse.csr_matrix) -> Recommender: ...

    def score(self, user: int, interactions: sparse.csr_matrix) -> Scores:
        """A score for every item; higher is better. Seen items are filtered by the caller."""
        ...


def top_k(scores: Scores, seen: npt.NDArray[np.int32], k: int) -> npt.NDArray[np.int64]:
    scores = scores.astype(np.float64, copy=True)
    scores[seen] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return np.array([], dtype=np.int64)
    candidates = np.argpartition(-scores, k - 1)[:k]
    return candidates[np.argsort(-scores[candidates], kind="stable")].astype(np.int64)


def zscore(x: Scores) -> Scores:
    std = float(x.std())
    return ((x - x.mean()) / std).astype(np.float32) if std > 0 else np.zeros_like(x)
