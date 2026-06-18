"""Text embedding engine — lightweight wrapper for BERT-based embedding models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default model: 33M params, 384-dim, ~67 MB download
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Lazy-loading wrapper for a text embedding model.

    Uses ``transformers`` + ``torch`` to load a BERT-based embedding model
    (e.g. BGE-small) and produce L2-normalised embeddings as numpy arrays.

    The model is loaded on first use, not at construction time.
    """

    def __init__(self, model_path: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_path = model_path
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s", self.model_path)
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModel.from_pretrained(self.model_path)
        self._model.eval()
        logger.info("Embedding model loaded successfully")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into L2-normalised embeddings.

        Returns an ``(N, D)`` float32 numpy array where *D* is the model's
        hidden dimension (384 for BGE-small).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            attention_mask = inputs["attention_mask"]

            # Mean pooling over token embeddings
            hidden = outputs.last_hidden_state  # (N, seq_len, D)
            mask_expanded = attention_mask.unsqueeze(-1)  # (N, seq_len, 1)
            summed = (hidden * mask_expanded).sum(dim=1)  # (N, D)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (N, 1)
            embeddings = summed / counts  # (N, D)

            # L2 normalise so dot product == cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy().astype(np.float32)
