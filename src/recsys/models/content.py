"""Content-based and hybrid models.

Collaborative filtering knows nothing about an item until people interact with
it, and little about a user with two ratings. Content embeddings (title, genres,
tags) fill that gap; the hybrid blends both and leans on content when the
user's history is short.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import sparse

from recsys.models.base import Recommender, Scores, zscore


class ContentBased:
    """User profile = mean embedding of liked items; score = cosine to each item."""

    name = "content"

    def __init__(self, item_embeddings: npt.NDArray[np.float32]) -> None:
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (item_embeddings / norms).astype(np.float32)

    def fit(self, interactions: sparse.csr_matrix) -> ContentBased:
        return self

    def score(self, user: int, interactions: sparse.csr_matrix) -> Scores:
        liked = interactions[user].indices
        if len(liked) == 0:
            return np.zeros(len(self.embeddings), dtype=np.float32)
        profile = self.embeddings[liked].mean(axis=0)
        scores: Scores = (self.embeddings @ profile).astype(np.float32)
        return scores


class Hybrid:
    """Blend of z-scored collaborative and content scores.

    The content weight decays with history length: ``w = 1 / (1 + n / half_life)``
    where ``n`` is how many items the user has interacted with. New users get
    mostly content; established users get mostly collaborative filtering.
    """

    name = "hybrid"

    def __init__(self, collaborative: Recommender, content: ContentBased, half_life: float = 10.0):
        self.collaborative = collaborative
        self.content = content
        self.half_life = half_life

    def fit(self, interactions: sparse.csr_matrix) -> Hybrid:
        self.collaborative.fit(interactions)
        self.content.fit(interactions)
        return self

    def content_weight(self, history: int) -> float:
        return 1.0 / (1.0 + history / self.half_life)

    def score(self, user: int, interactions: sparse.csr_matrix) -> Scores:
        w = self.content_weight(interactions[user].nnz)
        cf = zscore(self.collaborative.score(user, interactions))
        cb = zscore(self.content.score(user, interactions))
        return ((1 - w) * cf + w * cb).astype(np.float32)


def genre_features(genres: list[str]) -> npt.NDArray[np.float32]:
    """Multi-hot genre vectors: a dependency-free content representation."""
    vocab = sorted({g for row in genres for g in row.split("|") if g != "(no genres listed)"})
    position = {g: i for i, g in enumerate(vocab)}
    out = np.zeros((len(genres), len(vocab)), dtype=np.float32)
    for r, row in enumerate(genres):
        for g in row.split("|"):
            if g in position:
                out[r, position[g]] = 1.0
    return out
