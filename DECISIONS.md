# DECISIONS.md

Non-obvious choices and deviations, with reasons.

## Corpus

20 Newsgroups via sklearn `fetch_20newsgroups` (downloads on first use, no scraping). Train split is the indexed corpus, test split is the query source, so queries cannot leak into the index.

- Passages: document bodies with headers, footers and quotes removed, 150 per topic, 3000 total. Bodies shorter than 40 characters are skipped, bodies truncated at 1200 characters. Sampling is seeded (13) per topic so the build is deterministic.
- Queries: 80 subject lines from test-split documents, 4 per topic, deduped, minimum 4 words after stripping Re:/Fwd: prefixes. The spec suggested "titles or summaries rewritten into questions"; subjects are the natural titles here and rule-based question rewriting without a language model would only distort them, so they are used verbatim.

## Relevance model and metrics

Every corpus passage of the query's topic counts as relevant (R = topic passage set).

- Recall@1 = 1 if the top-1 passage is in R.
- Recall@5 = |top5 n R| / min(5, |R|), that is topic recall of the top 5 rather than recall of the whole topic set (each topic has 150 passages, so full-set recall would be capped near 0.03 and carry no signal).
- Hit@5 = 1 if any top-5 passage is in R.

## Tree

k-means (not agglomerative): faster on 3000 points, deterministic with a fixed seed, and good enough for a first pass. 20 top clusters matches the 20 ground-truth topics but is chosen unsupervised; the tree never sees newsgroup labels. Second level splits each top cluster into subclusters of at most 60 passages, minimum 2, so every level-2 decision is a real branch choice. Cluster centroids are arithmetic means of member embeddings. With unit vectors, dot(query, centroid) equals the mean cosine similarity over members exactly, so the spec's "mean member similarity" scoring costs one vector comparison per child.

## Embeddings

all-MiniLM-L6-v2 on CPU via sentence-transformers, L2 normalized. Fallback TF-IDF plus truncated SVD (128 dims) exists in `src/embeddings.py` and is selected automatically if the model fails to load; the first build on this machine actually hit the fallback due to a torch DLL initialization failure (see below), after the fix MiniLM is used and recorded in `data/meta.json` as `minilm`.

## Windows specifics

- torch on this machine fails to initialize (WinError 1114 on c10.dll) if scipy/sklearn loads its BLAS first. Fixed by importing torch before any sklearn import in `run.py`. Documented in README.
- The spec targets Python 3.11+; the interpreter behind `python` on this machine is 3.10.11, and the definition of done requires `python run.py` to run end to end, so the code uses only 3.10-compatible syntax (a subset of 3.11+) and runs unchanged on 3.11+.
- Upgrading sentence-transformers required upgrading torch (2.6 to 2.14) and torchvision in the user site, since transformers 5.x needs a torch newer than 2.6.

## Known simplifications

- `nav` returns at most k passages from one subtree; a subtree smaller than k returns fewer results and no backtracking or sibling expansion exists. A pruned branch is lost forever, which is the failure mode under study.
- Vectors compared per query counts centroid and passage comparisons together; k-means build cost (66 s including embedding 3000 passages) is recorded in `data/meta.json` as `build_seconds`.
- nav descends by argmax over child centroids with no beam, no tie handling beyond numpy argmax order.
