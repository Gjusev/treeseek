"""Embedding backends.

Primary: all-MiniLM-L6-v2 via sentence-transformers on CPU.
Fallback: TF-IDF plus truncated SVD (LSA) fitted on the corpus passages.
Both return L2-normalized row vectors.
"""
import numpy as np

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LSA_DIM = 128


def normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


class MiniLMEncoder:
    kind = "minilm"
    label = MINILM_NAME

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MINILM_NAME, device="cpu")

    def encode(self, texts):
        return normalize(self.model.encode(list(texts), show_progress_bar=False))


class LSAEncoder:
    kind = "lsa"
    label = f"tfidf+truncated-svd-{LSA_DIM}"

    def __init__(self, corpus_texts, seed=13):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(min_df=2, stop_words="english", max_features=50000)
        X = self.vec.fit_transform(list(corpus_texts))
        k = min(LSA_DIM, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=max(k, 2), random_state=seed)
        self.svd.fit(X)

    def encode(self, texts):
        return normalize(self.svd.transform(self.vec.transform(list(texts))))


def get_encoder(corpus_texts):
    """Try MiniLM, fall back to LSA if the model cannot be loaded or run."""
    try:
        enc = MiniLMEncoder()
        if enc.encode(["ping test"]).shape[1] > 0:
            return enc
    except Exception:
        pass
    return LSAEncoder(corpus_texts)
