"""Offline unit tests on synthetic vectors. No network, no model download."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import clean_subject
from src.embeddings import normalize
from src.eval import metrics, topic_sets
from src.retrieval import FlatRetriever, NavRetriever
from src.tree import build_tree


def synthetic(n_per_topic=12, dim=8, seed=7):
    """Three well-separated topic blobs, unit-normalized."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim)) * 3
    X, labels = [], []
    for c in range(3):
        pts = centers[c] + rng.normal(scale=0.2, size=(n_per_topic, dim))
        X.append(pts)
        labels.extend([c] * n_per_topic)
    return normalize(np.vstack(X)), np.asarray(labels)


def build_retrievers(X):
    tree = build_tree(X, n_top=3, leaf_max=5, seed=7)
    flat = FlatRetriever(X)
    nav = NavRetriever(
        tree["l1_centroids"],
        tree["l2_centroids"],
        tree["l2_parent"],
        tree["l2_members"],
        X,
    )
    return tree, flat, nav


def test_flat_matches_brute_force():
    X, _ = synthetic()
    _, flat, _ = build_retrievers(X)
    q = X[0]
    ranked, compared, _ = flat.search(q, k=3)
    brute = list(np.argsort(-(X @ q))[:3])
    assert ranked == [int(i) for i in brute]
    assert compared == len(X)


def test_nav_returns_only_chosen_subtree():
    X, _ = synthetic()
    tree, _, nav = build_retrievers(X)
    q = X[5]
    ranked, compared, path = nav.search(q, k=5)
    subtree = set(tree["l2_members"][path["l2"]].tolist())
    assert set(ranked).issubset(subtree)
    # compared: all L1 children + chosen L1's children + subtree passages
    expected = (
        len(tree["l1_centroids"])
        + int((tree["l2_parent"] == path["l1"]).sum())
        + path["subtree_size"]
    )
    assert compared == expected
    assert compared < len(X)  # the point of navigation


def test_nav_far_query_still_descends():
    # a query far from every passage still descends; a subtree smaller than k
    # yields fewer than k results, which is inherent to subtree-only return
    X, _ = synthetic()
    _, _, nav = build_retrievers(X)
    rng = np.random.default_rng(0)
    q = normalize(rng.normal(size=(1, X.shape[1])))[0]
    ranked, _, path = nav.search(q, k=4)
    assert 1 <= len(ranked) <= 4
    assert len(ranked) == min(4, path["subtree_size"])


def test_metrics_definitions():
    relevant = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    m = metrics([1, 20, 2, 30, 40], relevant, k=5)
    assert m["recall@1"] == 1.0
    assert m["recall@5"] == 2 / 5
    assert m["hit@5"] == 1.0
    m2 = metrics([99, 20, 2, 30, 40], relevant, k=5)
    assert m2["recall@1"] == 0.0
    assert m2["hit@5"] == 1.0
    # |R| < k: recall is against min(k, |R|)
    m3 = metrics([50, 1], {1}, k=5)
    assert m3["recall@5"] == 1 / 1
    assert m3["hit@5"] == 1.0


def test_topic_sets():
    sets = topic_sets(
        [{"id": 0, "topic": "a"}, {"id": 1, "topic": "b"}, {"id": 2, "topic": "a"}]
    )
    assert sets == {"a": {0, 2}, "b": {1}}


def test_clean_subject():
    assert clean_subject("Re: this is a subject") == "this is a subject"
    assert clean_subject("  Fwd:   multiple   spaces ") == "multiple spaces"
