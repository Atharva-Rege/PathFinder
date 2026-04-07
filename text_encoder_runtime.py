from __future__ import annotations

from typing import Any


def build_text_encoder(model_name: str) -> Any:
    """
    Build a sentence encoder using sentence-transformers only.
    """
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load sentence-transformers model for runtime text encoding. "
            "Ensure internet/cache availability and correct model name. "
            f"Original error: {exc}"
        )
