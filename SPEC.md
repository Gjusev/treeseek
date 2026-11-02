# SPEC: treeseek

You are developing this project autonomously. Read this spec fully, make sensible decisions without asking questions, and record every non-obvious choice in DECISIONS.md. Do not modify anything outside this directory.

## What this is

A minimal retrieval system that navigates a topic hierarchy from root to passage instead of scanning the whole corpus, compared against a flat dense retrieval baseline on the same corpus. It explores the idea in "Retriever-Free Retrieval: Navigating Documents with Decision Models" (method: DecisionRAG) (arXiv 2610.35868, Cornell + ASAPP): https://arxiv.org/abs/2610.35868

Key numbers from that paper, for context only: their trained decision model beats a dense retriever plus an 8B reranker by 13.8 Recall@1 on MultiDoc2Dual and 17.4 Hit@1 on Natural Questions over 2.68M paragraphs, at 0.2 to 2.3 cents per query, with no task-specific training.

Your version is a zero-training approximation: greedy descent through a corpus topic tree using embedding similarity, versus flat dense search. It is NOT a reproduction of their trained decision model and must never be described as one. The interesting question is whether plain hierarchical navigation already helps on a structured corpus, and where it loses (a branch pruned early is lost forever).

## What to build

1. Corpus: a small bundled corpus with natural topic structure. Good option: 20 Newsgroups subset (a few thousand passages, 20 topics) or another offline-loadable set. No scraping. Document the choice.
2. Build a two or three level tree: topics to subtopics to passages, built by clustering embeddings (agglomerative or simple k-means per level). Persist the tree.
3. Two retrievers behind one interface:
   - `flat`: cosine similarity over all passage embeddings, top-k.
   - `nav`: embed query, at each level score child clusters by mean or max member similarity, pick the best branch, descend, return top-k passages from the chosen subtree.
4. Eval: a fixed query set (at least 50 queries) with known relevant topics, derived from the corpus itself, for example hold-out passage titles or summaries rewritten into questions. Metrics: Recall@1, Recall@5, Hit@5, embeddings computed per query, and index build cost.

## Model or embeddings

Embeddings must run offline or cheaply. Preferred: a small local sentence-transformers model if installable on CPU (for example all-MiniLM or multilingual-MiniLM). Fallback: hashing-based or LSA embeddings from scikit-learn if transformer install fails. No paid API required. Document which was used, since it changes the meaning of the numbers.

## Layout

```
README.md
SPEC.md (this file, keep unchanged)
DECISIONS.md
src/
data/ (built artifacts: tree, index; not raw corpus if license forbids redistribution)
results/results.csv
RESULTS.md
run.py (flags: --build, --eval, --retriever flat|nav|both)
tests/
LICENSE (MIT)
```

## RESULTS.md

One table: rows = retrievers (flat, nav), columns = Recall@1, Recall@5, Hit@5, vectors compared per query. One short paragraph on when navigation won and when it lost, with a concrete failure case of early pruning if one occurred. One paragraph on limitations and the difference versus the trained decision model in the paper.

## Style rules

- README opens with: "Zero-training hierarchical navigation retrieval, an approximation of the navigation idea in arXiv 2610.35868."
- Cite the paper. State clearly there is no trained decision model here.
- No em dashes anywhere. English only. Engineer tone.
- Python 3.11+, stdlib plus numpy, scikit-learn, sentence-transformers if it installs cleanly. Keep dependencies minimal.

## Definition of done

- `python run.py --build` then `python run.py --eval --retriever both` runs end to end on CPU.
- RESULTS.md has a measured table.
- `pytest` passes.
- git repo initialized with meaningful commits.
