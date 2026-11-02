# nav-rag

Flat dense retrieval scores 3000 vectors per query. A two level topic tree reaches the same top-rank topic accuracy with 74. The cost is Hit@5: greedy descent prunes branches forever. Measured, one run on CPU:

| retriever | Recall@1 | Recall@5 | Hit@5 | vectors compared per query |
|---|---|---|---|---|
| flat | 0.613 | 0.518 | 0.812 | 3000 |
| nav | 0.650 | 0.523 | 0.750 | 74 |

Full numbers, per-query analysis and build cost are in [RESULTS.md](RESULTS.md).

## What it is

nav-rag is a small measured comparison of hierarchical navigation retrieval against flat dense retrieval, based on the navigation idea in "DecisionRAG: Retriever-Free Retrieval via Hierarchical Decision Models" (arXiv 2610.35868, Cornell and ASAPP, https://arxiv.org/abs/2610.35868). It is a zero-training approximation: branches are scored by mean member similarity of k-means centroids, not by the paper's trained hierarchical decision model, so this is not a reproduction and none of the numbers here say anything about the paper's numbers.

## Quickstart

```
python -m pip install -r requirements.txt
python run.py --build
python run.py --eval --retriever both
```

The first `--build` downloads the 20 Newsgroups corpus (sklearn) and the all-MiniLM-L6-v2 model (Hugging Face, about 90 MB), then embeds 3000 passages and builds the tree in about 66 s on CPU. `--eval` writes `results/results.csv` and `results/summary.json` and prints the table. `--retriever` takes `flat`, `nav`, or `both`. Tests: `python -m pytest`.

Windows note: torch DLLs must initialize before scipy loads its BLAS, so `run.py` imports torch first. If you import the modules from your own entry point, do the same.

## Corpus and method

Corpus: 3000 passages from the 20 Newsgroups train split, 150 per topic, headers, footers and quotes removed. Queries: 80 subject lines from held-out test split documents, 4 per topic, so no query text is in the index. Embeddings: all-MiniLM-L6-v2 on CPU, L2 normalized (a TF-IDF plus SVD fallback exists; the encoder actually used is recorded in `data/meta.json`). Relevance: every passage of the query's topic counts as relevant. Metric definitions are in [DECISIONS.md](DECISIONS.md).

k-means clusters the embeddings into a two level tree, root to 20 top clusters to 61 subclusters to passages (mean subtree size 49). The tree is built unsupervised, it never sees newsgroup labels. `flat` scores all 3000 passage vectors. `nav` scores the 20 top cluster centroids, descends into the argmax branch, scores that cluster's subcluster centroids, descends again, and returns top-k from the chosen subtree only. Because centroids are means of unit vectors, one dot product against a centroid equals the mean member cosine exactly, so each branch decision costs one vector comparison.

```
query embedding
      |
      v
20 top cluster centroids ------------------> argmax: top cluster 11
      |                                        (19 clusters never touched)
      v
subcluster centroids of cluster 11 ---------> argmax: subcluster 35
      |                                        (~55 other subclusters never touched)
      v
about 50 passage vectors of subcluster 35 --> top-k results
```

## The failure mode: early pruning

A branch not taken is never revisited. Hit@5 is where nav loses: flat finds at least one on-topic passage in the top 5 for 65 of 80 queries, nav for 60, with 57 solved by both, 3 by nav only and 8 by flat only. All 8 flat-only hits are pruned branches.

Concrete case: the query "Age of Reason Was: Who has read Rushdie's" (topic alt.atheism). flat answers it with two alt.atheism passages in the top 5. nav chose subcluster 35 under top cluster 11, a mixed subtree of sci.med, rec.sport.baseball, talk.politics.misc and talk.religion.misc passages, and since nav searches only that subtree, no alt.atheism passage is reachable at any rank. A related case: "Best Radar Detector - VALENTINE-1?" (rec.autos), where nav descended into a misc.forsale and sci.electronics subtree, semantically adjacent but without the correct topic.

## Limitations

- Zero-training approximation of a trained model. nav scores k-means centroids; DecisionRAG trains a hierarchical decision model over 2.68M paragraphs and reports gains over a dense retriever plus an 8B reranker. The two are not comparable.
- The corpus is small (3000 passages) and has clean topic structure that favors clustering. Queries are short subject lines, not natural questions. Relevance is topic-level.
- Greedy descent is irreversible: no backtracking, no beam, no sibling expansion. One wrong centroid comparison at level 1 removes about 1/20 of the corpus, which is exactly the 8 flat-only failures above.
- nav returns at most k passages from one subtree; a subtree smaller than k returns fewer results.
- Whether pruning losses grow with tree depth or corpus size, and whether a trained decision model would avoid these branch errors, are open questions this approximation does not answer. What it does answer: on a structured small corpus, plain hierarchical navigation already matches flat dense retrieval on top-rank topic accuracy at 1/40 of the comparison cost, while losing about 6 points of Hit@5 to irreversible pruning.

## Repository layout

```
run.py             CLI: --build, --eval, --retriever flat|nav|both
src/corpus.py      20 Newsgroups passages and held-out subject queries
src/embeddings.py  MiniLM encoder with LSA fallback
src/tree.py        two level k-means tree, centroid persistence
src/retrieval.py   flat and nav retrievers, one interface
src/eval.py        metrics: Recall@1, Recall@5, Hit@5
data/              built artifacts (corpus, queries, embeddings, tree, meta)
results/           results.csv, summary.json
tests/             offline unit tests on synthetic vectors
```

## License

MIT, see [LICENSE](LICENSE).

## Citation

The source paper this experiment is based on:

```bibtex
@misc{decisionrag2026,
  title         = {DecisionRAG: Retriever-Free Retrieval via Hierarchical Decision Models},
  year          = {2026},
  eprint        = {2610.35868},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.35868},
  note          = {Cornell University and ASAPP}
}
```

This repository:

```bibtex
@misc{navrag2026,
  title  = {nav-rag: hierarchical navigation retrieval vs flat dense retrieval on 20 Newsgroups},
  year   = {2026},
  note   = {Zero-training approximation of the navigation idea in arXiv:2610.35868, not a reproduction}
}
```
