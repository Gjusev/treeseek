"""nav-rag entry point.

Usage:
  python run.py --build
  python run.py --eval --retriever both
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:  # Windows: torch DLLs must initialize before scipy/sklearn loads its BLAS
    import torch  # noqa: F401
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import embeddings as emb_mod
from src.corpus import load_passages, load_queries
from src.eval import evaluate_retriever
from src.retrieval import FlatRetriever, NavRetriever
from src.tree import build_tree

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def build(seed: int):
    DATA.mkdir(exist_ok=True)
    t0 = time.time()
    passages = load_passages()
    queries = load_queries()
    enc = emb_mod.get_encoder([p["text"] for p in passages])
    emb = enc.encode([p["text"] for p in passages])
    tree = build_tree(emb, seed=seed)
    build_seconds = time.time() - t0

    np.savez_compressed(
        DATA / "embeddings.npz",
        passage_emb=emb,
        l1_centroids=tree["l1_centroids"],
        l2_centroids=tree["l2_centroids"],
        l2_parent=tree["l2_parent"],
    )
    (DATA / "corpus.json").write_text(
        json.dumps(
            {
                "passages": passages,
                "l1_labels": tree["l1_labels"].tolist(),
                "l2_labels": tree["l2_labels"].tolist(),
            }
        ),
        encoding="utf-8",
    )
    (DATA / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    meta = {
        "encoder": enc.kind,
        "encoder_label": enc.label,
        "n_passages": len(passages),
        "n_queries": len(queries),
        "n_l1_clusters": len(tree["l1_centroids"]),
        "n_l2_clusters": len(tree["l2_centroids"]),
        "subtree_size_mean": float(np.mean([len(m) for m in tree["l2_members"]])),
        "tree_params": tree["params"],
        "build_seconds": round(build_seconds, 1),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def _load_artifacts():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    corpus = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    z = np.load(DATA / "embeddings.npz")
    return meta, corpus, queries, z


def evaluate(retriever_names, k):
    RESULTS.mkdir(exist_ok=True)
    meta, corpus, queries, z = _load_artifacts()
    passages = corpus["passages"]
    enc = emb_mod.get_encoder([p["text"] for p in passages])
    q_emb = enc.encode([q["text"] for q in queries])

    retrievers = []
    if "flat" in retriever_names:
        retrievers.append(FlatRetriever(z["passage_emb"]))
    if "nav" in retriever_names:
        retrievers.append(
            NavRetriever(
                z["l1_centroids"],
                z["l2_centroids"],
                z["l2_parent"],
                [
                    np.flatnonzero(np.asarray(corpus["l2_labels"]) == j)
                    for j in range(len(z["l2_centroids"]))
                ],
                z["passage_emb"],
            )
        )

    all_rows, summaries = [], []
    for r in retrievers:
        rows, summary = evaluate_retriever(r, queries, q_emb, passages, k)
        all_rows.extend(rows)
        summaries.append(summary)

    # early-pruning cases: flat finds the topic in top-5, nav does not
    flat_rows = {r["query_id"]: r for r in all_rows if r["retriever"] == "flat"}
    nav_rows = {r["query_id"]: r for r in all_rows if r["retriever"] == "nav"}
    pruned = [
        {
            "query": n["query"],
            "topic": n["topic"],
            "flat_top5_topics": flat_rows[qid]["ranked_topics"],
            "nav_top5_topics": n["ranked_topics"],
            "nav_path": n["path"],
        }
        for qid, n in sorted(nav_rows.items())
        if qid in flat_rows
        and flat_rows[qid][f"hit@{k}"] == 1.0
        and n[f"hit@{k}"] == 0.0
    ]

    cols = [
        "query_id",
        "retriever",
        "topic",
        "top1_topic",
        "recall@1",
        f"recall@{k}",
        f"hit@{k}",
        "vectors_compared",
        "query",
    ]
    with open(RESULTS / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    out = {
        "meta": meta,
        "k": k,
        "summaries": summaries,
        "early_pruning_cases": pruned[:10],
        "n_early_pruning_cases": len(pruned),
    }
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    header = f"| retriever | Recall@1 | Recall@{k} | Hit@{k} | vectors compared per query |"
    print(header)
    print("|---|---|---|---|---|")
    for s in summaries:
        print(
            f"| {s['retriever']} | {s['recall@1']:.3f} | {s[f'recall@{k}']:.3f} "
            f"| {s[f'hit@{k}']:.3f} | {s['vectors_compared_mean']:.0f} |"
        )
    print(f"early pruning cases (flat hit, nav miss): {len(pruned)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="download corpus, embed, build tree")
    ap.add_argument("--eval", action="store_true", help="run evaluation")
    ap.add_argument("--retriever", choices=["flat", "nav", "both"], default="both")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if args.build:
        build(args.seed)
    if args.eval:
        names = {"flat", "nav"} if args.retriever == "both" else {args.retriever}
        evaluate(names, args.k)
    if not args.build and not args.eval:
        ap.print_help()


if __name__ == "__main__":
    main()
