"""Collaborative-filtering models: popularity, item-kNN and implicit ALS (from scratch)."""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from recsys.models.base import Scores

log = logging.getLogger(__name__)


class Popularity:
    """The baseline every recommender must beat, and the cold-start fallback."""

    name = "popularity"

    def fit(self, interactions: sparse.csr_matrix) -> Popularity:
        self.counts = np.asarray(interactions.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, user: int, interactions: sparse.csr_matrix) -> Scores:
        return self.counts


class ItemKNN:
    """Item-item cosine similarity, keeping the ``neighbours`` strongest links per item.

    Pruning keeps the similarity matrix sparse and removes the long tail of weak,
    noisy co-occurrences, which usually improves accuracy as well as speed.
    """

    name = "item_knn"

    def __init__(self, neighbours: int = 100, shrinkage: float = 10.0) -> None:
        self.neighbours = neighbours
        self.shrinkage = shrinkage

    def fit(self, interactions: sparse.csr_matrix) -> ItemKNN:
        X = interactions.T.tocsr().astype(np.float32)
        co = (X @ X.T).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrinkage)
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < sim.shape[0]:
            cut = np.argpartition(-sim, self.neighbours, axis=1)[:, self.neighbours :]
            np.put_along_axis(sim, cut, 0.0, axis=1)
        self.similarity = sparse.csr_matrix(sim.astype(np.float32))
        return self

    def score(self, user: int, interactions: sparse.csr_matrix) -> Scores:
        history = interactions[user]
        scores: Scores = np.asarray((history @ self.similarity).todense()).ravel()
        return scores.astype(np.float32)


class ImplicitALS:
    """Alternating least squares for implicit feedback (Hu, Koren & Volinsky, 2008).

    Every unobserved (user, item) pair is a weak negative with confidence 1; an
    observed one is a positive with confidence ``1 + alpha``. Each half-step is a
    closed-form ridge regression, using the trick that ``Y^T C_u Y = Y^T Y +
    Y^T (C_u - I) Y`` so only the user's observed items need to be touched.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 20.0,
        iterations: int = 15,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, interactions: sparse.csr_matrix) -> ImplicitALS:
        rng = np.random.default_rng(self.seed)
        n_users, n_items = interactions.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors)).astype(np.float64)
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors)).astype(np.float64)
        Cui = interactions.tocsr().astype(np.float64)
        Ciu = Cui.T.tocsr()
        for it in range(self.iterations):
            self._solve(Cui, self.user_factors, self.item_factors)
            self._solve(Ciu, self.item_factors, self.user_factors)
            log.debug("ALS iteration %d", it + 1)
        return self

    def _solve(self, C: sparse.csr_matrix, X: np.ndarray, Y: np.ndarray) -> None:
        YtY = Y.T @ Y
        eye = self.regularization * np.eye(self.factors)
        for u in range(C.shape[0]):
            start, end = C.indptr[u], C.indptr[u + 1]
            if start == end:
                X[u] = 0.0
                continue
            idx = C.indices[start:end]
            confidence = 1.0 + self.alpha * C.data[start:end]
            Yi = Y[idx]
            A = YtY + (Yi.T * (confidence - 1.0)) @ Yi + eye
            b = Yi.T @ confidence
            X[u] = np.linalg.solve(A, b)

    def score(self, user: int, interactions: sparse.csr_matrix) -> Scores:
        scores: Scores = (self.item_factors @ self.user_factors[user]).astype(np.float32)
        return scores

    def fold_in(self, items: list[int]) -> Scores:
        """Scores for a user who is not in the training data (one ALS half-step).

        Lets the API serve someone who just liked a few movies without retraining.
        """
        if not items:
            return np.zeros(len(self.item_factors), dtype=np.float32)
        Y = self.item_factors
        Yi = Y[items]
        confidence = np.full(len(items), 1.0 + self.alpha)
        A = Y.T @ Y + (Yi.T * (confidence - 1.0)) @ Yi + self.regularization * np.eye(self.factors)
        user = np.linalg.solve(A, Yi.T @ confidence)
        scores: Scores = (Y @ user).astype(np.float32)
        return scores
