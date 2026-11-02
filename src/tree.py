"""Two-level topic tree over passage embeddings.

root -> L1 clusters (k-means, K_TOP) -> L2 clusters (k-means within each L1,
sized so each subtree stays around LEAF_MAX passages) -> passages.

Cluster centroids are arithmetic means of their member passage embeddings.
With unit-normalized query and passage vectors, dot(query, centroid) equals
the mean cosine similarity from the query to every member passage, so scoring
one centroid implements the "mean member similarity" rule from the spec at
one vector comparison per child.
"""
import numpy as np
from sklearn.cluster import KMeans

K_TOP = 20  # matches the 20 newsgroup topics, chosen unsupervised
LEAF_MAX = 60  # target subtree size; sets k for the L2 split


def _kmeans_labels(X, k, seed):
    k = max(1, min(k, len(X)))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    return km.fit_predict(X)


def build_tree(emb, n_top=K_TOP, leaf_max=LEAF_MAX, seed=13):
    """Cluster passages and return a serializable tree plus centroid arrays.

    Returns dict with:
      l1_labels: (n,) int L1 cluster per passage
      l2_labels: (n,) int L2 cluster id, unique within the whole corpus
      l1_centroids: (n_top_effective, d)
      l2_centroids: (m, d)
      l2_parent: (m,) int, L1 index of each L2 cluster
      l2_members: list of m int arrays, passage ids per L2 cluster
    """
    n = len(emb)
    if n < n_top:
        raise ValueError(f"need at least {n_top} passages, got {n}")
    l1_labels = _kmeans_labels(emb, n_top, seed)

    l2_labels = np.full(n, -1, dtype=int)
    l2_parent = []
    l2_members = []
    for c in range(l1_labels.max() + 1):
        members = np.flatnonzero(l1_labels == c)
        if len(members) == 0:
            continue
        # at least 2 subclusters so level 2 is a real branch decision
        k2 = max(2, int(np.ceil(len(members) / leaf_max)))
        sub = _kmeans_labels(emb[members], k2, seed)
        for s in range(sub.max() + 1):
            l2_parent.append(c)
            l2_members.append(members[sub == s])
            l2_labels[members[sub == s]] = len(l2_parent) - 1

    l1_centroids = np.vstack(
        [emb[l1_labels == c].mean(axis=0) for c in range(l1_labels.max() + 1)]
    )
    l2_centroids = np.vstack([emb[m].mean(axis=0) for m in l2_members])
    return {
        "l1_labels": l1_labels,
        "l2_labels": l2_labels,
        "l1_centroids": l1_centroids.astype(np.float32),
        "l2_centroids": l2_centroids.astype(np.float32),
        "l2_parent": np.asarray(l2_parent, dtype=int),
        "l2_members": [np.asarray(m, dtype=int) for m in l2_members],
        "params": {"n_top": int(n_top), "leaf_max": int(leaf_max), "seed": int(seed)},
    }
