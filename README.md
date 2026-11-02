# nav-rag

Zero-training hierarchical navigation retrieval, an approximation of the navigation idea in arXiv 2610.35868.

This project explores the navigation idea from "DecisionRAG: Retriever-Free Retrieval via Hierarchical Decision Models" (arXiv 2610.35868, Cornell + ASAPP: https://arxiv.org/abs/2610.35868). The paper trains a decision model that walks a hierarchy from root to passage over 2.68M paragraphs and beats a dense retriever plus an 8B reranker. There is no trained decision model here. This repository contains only a zero-training approximation: greedy descent through a corpus topic tree using embedding similarity, compared against flat dense retrieval on the same corpus. It is not a reproduction of the paper's system and the numbers below say nothing about the paper's numbers.

## What it does

1. Builds a corpus of 3000 passages from the 20 Newsgroups train split (150 per topic, headers, footers and quotes removed) and a fixed set of 80 queries from subject lines of held-out test-split documents (4 per topic).
2. Embeds passages locally on CPU with all-MiniLM-L6-v2 (sentence-transformers). If the model cannot load, it falls back to TF-IDF plus truncated SVD (LSA, 128 dims). The encoder actually used is recorded in `data/meta.json`.
3. Clusters the embeddings with k-means into a two level tree: root to 20 top clusters to 61 subclusters to passages (mean subtree size 49).
4. Runs two retrievers behind one interface:
   - `flat`: cosine similarity over all 3000 passage embeddings, top-k.
   - `nav`: embed the query, at each level score child clusters by mean member similarity (one dot product against each child centroid), take the best branch, descend, and return top-k from the chosen subtree only. A branch not taken is never revisited.

## Usage

```
python -m pip install -r requirements.txt
python run.py --build
python run.py --eval --retriever both
python -m pytest
```

`--build` downloads the corpus on first use (sklearn fetch_20newsgroups) and the embedding model (Hugging Face, about 90 MB), then writes `data/`. `--eval` reads `data/`, writes `results/results.csv` and `results/summary.json`, and prints the summary table. `--retriever` accepts `flat`, `nav`, or `both`. Measured results live in RESULTS.md.

Note for Windows: torch DLLs must initialize before scipy loads its BLAS, so `run.py` imports torch first. If you import the modules from your own entry point, do the same.

## Repository layout

```
run.py          CLI: --build, --eval, --retriever flat|nav|both
src/corpus.py   20 Newsgroups passages and held-out subject queries
src/embeddings.py  MiniLM encoder with LSA fallback
src/tree.py     two level k-means tree, centroid persistence
src/retrieval.py   flat and nav retrievers, one interface
src/eval.py     metrics: Recall@1, Recall@5, Hit@5
data/           built artifacts (corpus, queries, embeddings, tree, meta)
results/        results.csv, summary.json
tests/          offline unit tests on synthetic vectors
```

## License

MIT, see LICENSE.
