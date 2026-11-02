"""Corpus and query set from 20 Newsgroups.

Passages: train-split document bodies (headers, footers, quotes removed),
capped per topic. Queries: subject lines of held-out test-split documents.
The train/test split keeps queries out of the indexed corpus.
"""
import re

import numpy as np
from sklearn.datasets import fetch_20newsgroups

SEED = 13
DOCS_PER_TOPIC = 150  # corpus cap: 20 topics x 150 = 3000 passages
MIN_BODY_CHARS = 40
MAX_PASSAGE_CHARS = 1200
QUERIES_PER_TOPIC = 4  # 20 topics x 4 = 80 queries, spec asks for at least 50
MIN_QUERY_WORDS = 4


def clean_subject(subject: str) -> str:
    s = re.sub(r"^\s*(re|fwd|fw):\s*", "", subject, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def load_passages():
    """Deterministic per-topic sample of train-split bodies as passages."""
    ds = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    rng = np.random.default_rng(SEED)
    passages = []
    for topic_id, topic in enumerate(ds.target_names):
        idx = np.flatnonzero(ds.target == topic_id)
        rng.shuffle(idx)
        taken = 0
        for i in idx:
            if taken >= DOCS_PER_TOPIC:
                break
            body = ds.data[i].strip()
            if len(body) < MIN_BODY_CHARS:
                continue
            passages.append(
                {
                    "id": len(passages),
                    "topic": topic,
                    "text": body[:MAX_PASSAGE_CHARS],
                }
            )
            taken += 1
    return passages


def _subject_of(raw_text: str) -> str:
    m = re.search(r"^Subject:\s*(.+)$", raw_text, flags=re.M)
    return clean_subject(m.group(1)) if m else ""


def load_queries():
    """Subject-line queries from held-out test-split documents, deduped per topic."""
    ds = fetch_20newsgroups(subset="test", remove=("footers", "quotes"))
    rng = np.random.default_rng(SEED + 1)
    queries = []
    for topic_id, topic in enumerate(ds.target_names):
        idx = np.flatnonzero(ds.target == topic_id)
        rng.shuffle(idx)
        seen = set()
        taken = 0
        for i in idx:
            if taken >= QUERIES_PER_TOPIC:
                break
            subj = _subject_of(ds.data[i])
            if not subj or len(subj.split()) < MIN_QUERY_WORDS or subj.lower() in seen:
                continue
            seen.add(subj.lower())
            queries.append({"id": len(queries), "topic": topic, "text": subj})
            taken += 1
    return queries
