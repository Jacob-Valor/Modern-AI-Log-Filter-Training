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

_ENTITY_BOOST_TYPES = {"Indicator", "Malware", "Vulnerability"}


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

    def _load(self) -> None:
        """Load HuggingFace pipeline on first use."""
        from transformers import pipeline

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
        logger.info("NER model loaded")

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
            batch = texts[i : i + self.batch_size]
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

            # Strip leading ## from subword tokens
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
