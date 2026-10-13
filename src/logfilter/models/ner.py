"""
SecureBERT2.0-NER wrapper — Named Entity Recognition for cybersecurity logs.

Extracts IOCs (IPs, CVEs, malware names), organizations, systems, and
vulnerability references from normalized log text.

Model: cisco-ai/SecureBERT2.0-NER
Task:  token-classification (ModernBertForTokenClassification, 11 labels)

Supported entity labels:
  B-Indicator / I-Indicator  → IPs, domains, file hashes
  B-Malware   / I-Malware    → malware / exploit names
  B-Organization/I-Organization → companies, groups
  B-System    / I-System     → affected software / platforms
  B-Vulnerability/I-Vulnerability → CVE IDs, flaw descriptions
  O                           → outside token
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# transformers sets model_max_length to this sentinel when a tokenizer declares
# no finite limit; treating it as a real budget would disable truncation.
_UNBOUNDED_MODEL_MAX_LENGTH = int(1e30)


def _tokenizer_kwargs(pipeline_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract the from_pretrained kwargs the tokenizer also accepts."""
    return {k: v for k, v in pipeline_kwargs.items() if k in ("cache_dir", "revision")}


def _max_tokens(tokenizer: Any, model: Any = None) -> int:
    """Return the model's real token ceiling.

    Prefers the model's position-embedding table, which is the limit that
    actually raises when exceeded. Falls back to the tokenizer's declared
    ``model_max_length`` and finally to the class default, treating
    transformers' "unbounded" sentinel as no declaration at all.
    """
    candidates = [getattr(getattr(model, "config", None), "max_position_embeddings", None)]
    candidates.append(getattr(tokenizer, "model_max_length", None))
    candidates.append(NERModel.DEFAULT_MAX_TOKENS)
    for value in candidates:
        if isinstance(value, int) and 0 < value < _UNBOUNDED_MODEL_MAX_LENGTH:
            return value
    return NERModel.DEFAULT_MAX_TOKENS


def _max_input_chars(tokenizer: Any, model: Any = None) -> int:
    """Character budget that cannot exceed the model's token ceiling.

    Uses a conservative (high) chars-per-token factor so the text is always
    cut *before* it could exceed the limit; being conservative only drops a few
    extra tokens of a pathological log line, which is the intended trade.
    """
    return _max_tokens(tokenizer, model) * NERModel.CONSERVATIVE_CHARS_PER_TOKEN


@dataclass
class ExtractedEntities:
    """Parsed entities from a single log text."""

    indicators: list[str] = field(default_factory=list)  # IPs, domains, hashes
    malware: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)
    confidence: float = 0.0  # max confidence across all extracted entities
    has_high_value_entities: bool = False  # any Indicator/Malware/Vulnerability

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicators": self.indicators,
            "malware": self.malware,
            "vulnerabilities": self.vulnerabilities,
            "organizations": self.organizations,
            "systems": self.systems,
            "confidence": self.confidence,
            "has_high_value_entities": self.has_high_value_entities,
        }

    def flat_entity_string(self) -> str:
        """Comma-separated string of all entities for LEEF field."""
        all_ents = (
            self.indicators
            + self.malware
            + self.vulnerabilities
            + self.organizations
            + self.systems
        )
        return ",".join(dict.fromkeys(all_ents))  # deduplicated, order-preserving


class NERModel:
    """
    Lazy-loading wrapper around SecureBERT2.0-NER.

    The model is NOT loaded until the first call to extract() to keep
    startup time fast and allow CPU-only / GPU configuration at call time.
    """

    MODEL_ID = "cisco-ai/SecureBERT2.0-NER"

    # SecureBERT2.0 has an 8192-token context. Used when the tokenizer does not
    # declare a finite model_max_length.
    DEFAULT_MAX_TOKENS = 8192
    # Log text averages well under one token per character; 4 is a deliberately
    # conservative ceiling that keeps the byte budget under the token limit.
    CONSERVATIVE_CHARS_PER_TOKEN = 4
    TRUNCATION_MARKER = "…[truncated]"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        batch_size: int = 32,
        min_confidence: float = 0.80,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.revision = revision
        self._pipeline: Any | None = None
        self._tokenizer: Any | None = None
        default_budget = NERModel.DEFAULT_MAX_TOKENS * NERModel.CONSERVATIVE_CHARS_PER_TOKEN
        self._max_input_chars: int = default_budget

    def _load(self) -> None:
        """Load HuggingFace pipeline on first use."""
        from transformers import AutoTokenizer, pipeline

        logger.info("Loading NER model", model_id=self.model_id, device=self.device)
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "aggregation_strategy": "simple",
            "device": 0 if self.device == "cuda" else -1,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        if self.revision is not None:
            kwargs["revision"] = self.revision
        self._pipeline = pipeline("token-classification", **kwargs)

        # transformers v5's TokenClassificationPipeline._sanitize_parameters() does
        # not accept `truncation` (nor forward tokenizer kwargs), so an over-long
        # log line would raise instead of truncating and kill the whole batch.
        # Bound the text ourselves against the model's real ceiling instead. The
        # tokenizer alone is not enough: it reports 1e30 ("unbounded") whenever
        # no finite limit was declared, which disables its own truncation and
        # lets the text reach the position-embedding table and raise.
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **_tokenizer_kwargs(kwargs))
        model = getattr(self._pipeline, "model", None)
        self._tokenizer = tokenizer
        self._max_input_chars = _max_input_chars(tokenizer, model)
        logger.info(
            "NER model loaded",
            max_tokens=_max_tokens(tokenizer, model),
            max_input_chars=self._max_input_chars,
        )

    def _truncate(self, text: str) -> str:
        """Clamp *text* to the input budget so an over-long line cannot raise."""
        if len(text) <= self._max_input_chars:
            return text
        keep = self._max_input_chars - len(self.TRUNCATION_MARKER)
        logger.warning(
            "Truncating over-long text for NER",
            original_chars=len(text),
            kept_chars=keep,
            max_input_chars=self._max_input_chars,
        )
        return text[:keep] + self.TRUNCATION_MARKER

    def extract(self, text: str) -> ExtractedEntities:
        """Extract entities from a single text string."""
        results = self.extract_batch([text])
        return results[0]

    def extract_batch(self, texts: list[str]) -> list[ExtractedEntities]:
        """
        Extract entities from a batch of text strings.

        Returns a list of ExtractedEntities, one per input text.
        """
        if self._pipeline is None:
            self._load()

        results = []
        # Process in sub-batches
        for i in range(0, len(texts), self.batch_size):
            batch = [self._truncate(text) for text in texts[i : i + self.batch_size]]
            raw_batch = self._pipeline(batch)  # type: ignore[misc]
            # When given a list, HF pipeline returns list of list of dicts
            if texts and not isinstance(raw_batch[0], list):
                raw_batch = [raw_batch]  # single text fallback
            for raw_entities in raw_batch:
                results.append(self._parse_entities(raw_entities))

        return results

    def _parse_entities(self, raw_entities: list[dict]) -> ExtractedEntities:
        ent = ExtractedEntities()
        max_conf = 0.0

        for item in raw_entities:
            conf = item.get("score", 0.0)
            if conf < self.min_confidence:
                continue
            label = item.get("entity_group", item.get("entity", "O"))
            word = item.get("word", "").strip()
            if not word or label == "O":
                continue

            # Strip ALL leading hashes from subword tokens ("##word", "###").
            # lstrip strips a character SET, which is the intended behaviour here: a
            # token fragment may carry several word-piece markers, and removing only
            # one with removeprefix("#") left fragments like "##" and "#10.0.0.5" in
            # the indicator list.
            word = word.lstrip("#").strip()
            if not word:
                continue

            max_conf = max(max_conf, conf)

            if "Indicator" in label:
                ent.indicators.append(word)
                ent.has_high_value_entities = True
            elif "Malware" in label:
                ent.malware.append(word)
                ent.has_high_value_entities = True
            elif "Vulnerability" in label:
                ent.vulnerabilities.append(word)
                ent.has_high_value_entities = True
            elif "Organization" in label:
                ent.organizations.append(word)
            elif "System" in label:
                ent.systems.append(word)

        ent.confidence = max_conf
        return ent
