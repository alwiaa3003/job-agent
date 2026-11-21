"""Local sentence embeddings (must match the model that built data/job_vecs.npy)."""

import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"
DESC_CHARS = 1000

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer   # slow import, load lazily
        try:
            # Avoids a network round-trip to Hugging Face on every start (and hangs offline).
            _model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
        except OSError:
            _model = SentenceTransformer(EMBED_MODEL)
    return _model


def job_text(title: str, description: str) -> str:
    """The text embedded for one job; must stay identical to the one used for the saved vectors."""
    return f"{title}. {description[:DESC_CHARS]}"


def embed_text(text: str) -> np.ndarray:
    """Normalised 384-dim vector for one text."""
    vec = _get_model().encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Normalised vectors for many texts, shape (n, 384)."""
    vecs = _get_model().encode(list(texts), batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
    return vecs.astype(np.float32)
