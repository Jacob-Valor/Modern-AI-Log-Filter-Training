"""
SecureBERT2.0-CrossEncoder wrapper.

Computes pairwise relevance scores between a normalized log text and
a set of MITRE ATT&CK technique descriptions. Used in Tier 3 of the
inference pipeline after the BiEncoder has retrieved top-k candidates.

Model: cisco-ai/SecureBERT2.0-cross_encoder
Type:  CrossEncoder / Sentence Similarity (single logit; sigmoid applied here)
Max sequence length: 1024 tokens
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _identity_activation() -> Any:
    """Identity activation for ``CrossEncoder.predict``.

    This model has ``num_labels == 1``, for which sentence-transformers applies a
    ``Sigmoid`` by default, so ``predict`` already returns a probability. That
    default combined with the wrapper's own sigmoid squashed every score into
    ``(0.5, 0.731)``, which made this 0.40-weighted scoring term incapable of
    moving the composite score across a routing boundary. Requesting the
    identity guarantees exactly one sigmoid is applied below, independent of the
    installed library version.
    """
    from torch.nn import Identity

    return Identity()


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


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
        # Flatten into (text, description) pairs with bookkeeping
        all_pairs: list[tuple[str, str]] = []
        lengths: list[int] = []

        for text, candidates in zip(log_texts, candidates_per_log):
            pairs = [(text, c["description"]) for c in candidates]
            all_pairs.extend(pairs)
            lengths.append(len(pairs))

        # Nothing to score — return empty results without loading the model.
        # (Loading eagerly here would hit the network even for empty input.)
        if not all_pairs:
            return [[] for _ in log_texts]

        if self._model is None:
            self._load()

        # Batch prediction. ``activation_fn=_identity_activation()`` is load-bearing:
        # with the library default, ``predict`` returns a sigmoid'd probability for
        # this single-logit model, and the sigmoid below would be a second one.
        raw_scores = self._model.predict(  # type: ignore[union-attr]
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            activation_fn=_identity_activation(),
        )

        # Re-assemble per-log results
        results = []
        offset = 0
        for i, (length, candidates) in enumerate(zip(lengths, candidates_per_log)):
            log_scores = []
            for j, candidate in enumerate(candidates):
                raw = float(raw_scores[offset + j])
                # ``raw`` is a raw logit (see ``activation_fn`` above); this is the
                # single sigmoid that maps it onto 0–1.
                score_01 = _sigmoid(raw)
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
