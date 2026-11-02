"""Evaluation: metrics over a fixed query set with known relevant topics.

Relevance model: every corpus passage of the query's topic is relevant
(R = topic passage set). Metric definitions used here:
  Recall@1 = 1 if the top-1 passage is in R else 0
  Recall@5 = |top5 n R| / min(5, |R|)  (topic recall of the top 5)
  Hit@5    = 1 if any top-5 passage is in R else 0
"""
import numpy as np


def topic_sets(passages):
    """Map topic name -> set of passage ids."""
    sets = {}
    for p in passages:
        sets.setdefault(p["topic"], set()).add(p["id"])
    return sets


def metrics(ranked, relevant, k=5):
    top = list(ranked[:k])
    hits = sum(1 for i in top if i in relevant)
    return {
        "recall@1": 1.0 if ranked and ranked[0] in relevant else 0.0,
        f"recall@{k}": hits / min(k, len(relevant)) if relevant else 0.0,
        f"hit@{k}": 1.0 if hits > 0 else 0.0,
    }


def evaluate_retriever(retriever, queries, q_emb, passages, k=5):
    """Run one retriever over all queries, aggregate metrics, collect failures.

    A failure case is a query where flat-style full scan would have found the
    topic but navigation did not (nav Hit@5 = 0); we record it with the chosen
    path so early pruning can be inspected.
    """
    tsets = topic_sets(passages)
    topic_of = {p["id"]: p["topic"] for p in passages}
    rows = []
    for q, qv in zip(queries, q_emb):
        ranked, compared, path = retriever.search(qv, k)
        m = metrics(ranked, tsets[q["topic"]], k)
        rows.append(
            {
                "query_id": q["id"],
                "query": q["text"],
                "topic": q["topic"],
                "retriever": retriever.name,
                "top1_topic": topic_of.get(ranked[0]) if ranked else "",
                "ranked_topics": [topic_of.get(i, "") for i in ranked],
                **m,
                "vectors_compared": compared,
                "path": path,
            }
        )
    summary = {
        "retriever": retriever.name,
        "n_queries": len(rows),
        "recall@1": float(np.mean([r["recall@1"] for r in rows])) if rows else 0.0,
        f"recall@{k}": float(np.mean([r[f"recall@{k}"] for r in rows])) if rows else 0.0,
        f"hit@{k}": float(np.mean([r[f"hit@{k}"] for r in rows])) if rows else 0.0,
        "vectors_compared_mean": float(np.mean([r["vectors_compared"] for r in rows]))
        if rows
        else 0.0,
    }
    return rows, summary
