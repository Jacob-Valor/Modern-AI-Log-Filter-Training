"""
SecureBERT2.0-CrossEncoder wrapper.

Computes pairwise relevance scores between a normalized log text and
a set of MITRE ATT&CK technique descriptions. Used in Tier 3 of the
inference pipeline after the BiEncoder has retrieved top-k candidates.

Model: cisco-ai/SecureBERT2.0-cross_encoder
Type:  CrossEncoder / Sentence Similarity (output: scalar 0–1)
Max sequence length: 1024 tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CrossEncoderScore:
    technique_id: str
    name: str
    score: float  # 0–1 relevance to this ATT&CK technique


class CrossEncoderModel:
    """
    Lazy-loading wrapper around SecureBERT2.0-cross_encoder.

    Given a (log_text, technique_description) pair, returns a float score.
    Designed to run only on the top-k ATT&CK candidates retrieved by the
    BiEncoder, not against all 50+ techniques.
    """

    MODEL_ID = "cisco-ai/SecureBERT2.0-cross_encoder"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        batch_size: int = 16,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.revision = revision
        self._model: Any | None = None

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading CrossEncoder model", model_id=self.model_id)
        ce_kwargs: dict[str, Any] = {
            "device": self.device,
            "max_length": 1024,
        }
        if self.cache_dir is not None:
            ce_kwargs["cache_folder"] = str(self.cache_dir)
        if self.revision is not None:
            ce_kwargs["revision"] = self.revision
        self._model = CrossEncoder(self.model_id, **ce_kwargs)
        logger.info("CrossEncoder model loaded")

    def score(
        self,
        log_text: str,
        candidates: list[dict[str, str]],
    ) -> list[CrossEncoderScore]:
        """
        Score a single log text against a list of ATT&CK technique candidates.

        Parameters
        ----------
        log_text : str
            Normalized log text from the normalizer.
        candidates : list[dict]
            Each dict must have 'id', 'name', and 'description' keys.

        Returns
        -------
        list[CrossEncoderScore] sorted by score descending.
        """
        if not candidates:
            return []
        return self.score_batch([log_text], [candidates])[0]

    def score_batch(
        self,
        log_texts: list[str],
        candidates_per_log: list[list[dict[str, str]]],
    ) -> list[list[CrossEncoderScore]]:
        """
        Score multiple log texts against their respective ATT&CK candidates.

        Parameters
        ----------
        log_texts : list[str]
            One normalized text per log event.
        candidates_per_log : list[list[dict]]
            One list of candidate dicts per log event.

        Returns
        -------
        list[list[CrossEncoderScore]] — outer list matches log_texts order.
        """
        if self._model is None:
            self._load()

        # Flatten into (text, description) pairs with bookkeeping
        all_pairs: list[tuple[str, str]] = []
        lengths: list[int] = []

        for text, candidates in zip(log_texts, candidates_per_log):
            pairs = [(text, c["description"]) for c in candidates]
            all_pairs.extend(pairs)
            lengths.append(len(pairs))

        if not all_pairs:
            return [[] for _ in log_texts]

        # Batch prediction
        raw_scores = self._model.predict(  # type: ignore[union-attr]
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Re-assemble per-log results
        results = []
        offset = 0
        for i, (length, candidates) in enumerate(zip(lengths, candidates_per_log)):
            log_scores = []
            for j, candidate in enumerate(candidates):
                raw = float(raw_scores[offset + j])
                # CrossEncoder outputs a raw logit; apply sigmoid for 0–1 range
                score_01 = 1.0 / (1.0 + (2.718281828**-raw))
                log_scores.append(
                    CrossEncoderScore(
                        technique_id=candidate.get("id", ""),
                        name=candidate.get("name", ""),
                        score=score_01,
                    )
                )
            log_scores.sort(key=lambda x: x.score, reverse=True)
            results.append(log_scores)
            offset += length

        return results
