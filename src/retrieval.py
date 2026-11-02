"""Two retrievers behind one interface.

Both .search(q, k) return (ranked passage ids, vectors compared, path dict).
q must be a L2-normalized vector.
"""
import numpy as np


def _topk_ids(ids, sims, k):
    order = np.argsort(-sims)[:k]
    return [int(i) for i in np.asarray(ids)[order]]


class FlatRetriever:
    """Cosine similarity over all passage embeddings."""

    name = "flat"

    def __init__(self, emb):
        self.emb = emb

    def search(self, q, k=5):
        sims = self.emb @ q
        return _topk_ids(np.arange(len(self.emb)), sims, k), len(self.emb), {}


class NavRetriever:
    """Greedy root-to-leaf descent: best child centroid per level, then top-k
    inside the chosen subtree. A branch not taken is never revisited."""

    name = "nav"

    def __init__(self, l1_centroids, l2_centroids, l2_parent, l2_members, emb):
        self.l1_centroids = l1_centroids
        self.l2_centroids = l2_centroids
        self.l2_parent = np.asarray(l2_parent, dtype=int)
        self.l2_members = [np.asarray(m, dtype=int) for m in l2_members]
        self.emb = emb

    def search(self, q, k=5):
        compared = 0
        l1_sims = self.l1_centroids @ q
        l1 = int(np.argmax(l1_sims))
        compared += len(self.l1_centroids)

        children = np.flatnonzero(self.l2_parent == l1)
        l2 = int(children[np.argmax(self.l2_centroids[children] @ q)])
        compared += len(children)

        members = self.l2_members[l2]
        sims = self.emb[members] @ q
        compared += len(members)
        top = _topk_ids(members, sims, k)
        path = {"l1": l1, "l2": l2, "subtree_size": int(len(members))}
        return top, compared, path
