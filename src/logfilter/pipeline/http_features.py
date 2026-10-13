"""Label-blind lexical features for structured HTTP/WAF telemetry."""

from __future__ import annotations

import math
import re
import urllib.parse
from collections import Counter
from typing import Any

HTTP_FEATURE_NAMES = (
    "uri_length",
    "body_length",
    "query_parameter_count",
    "percent_escape_count",
    "malformed_percent_count",
    "decode_depth",
    "sql_keyword_count",
    "sql_comment_count",
    "sql_tautology_count",
    "xss_tag_count",
    "xss_handler_count",
    "javascript_scheme_count",
    "traversal_count",
    "sensitive_path_count",
    "nul_count",
    "shell_metacharacter_count",
    "max_token_length",
    "character_entropy",
)

_PERCENT_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
_MALFORMED_PERCENT_RE = re.compile(r"%(?![0-9a-fA-F]{2})")
_SQL_KEYWORD_RE = re.compile(
    # The ``select ... from`` branch is written so that no two adjacent quantifiers
    # can match the same character. An unbounded ``\s+.+?\s+`` is catastrophically
    # backtracking on attacker-controlled text (measured 2.7 s at 1.6 KB, growing
    # ~n**2.5, because every start position re-scanned a long whitespace run while
    # backtracking for "from"). Merely bounding the wildcard is NOT enough: the
    # adjacent ``\s+`` quantifiers still overlapped and the same blowup remained
    # (measured 160 s at 20 KB). The fix is to make each quantifier's character class
    # disjoint from its neighbour's - ``\s`` here never touches ``[\w,()*'"=-]`` -
    # and to bound every repetition, so each start position costs a fixed ~1 k ops.
    r"\b(?:union\s{1,32}(?:all\s{1,32})?select"
    r"|select\s{1,256}[\w,()*'\"=-]{0,512}\s{1,256}from"
    r"|insert\s{1,32}into"
    r"|update\s{1,32}\w{1,64}\s{1,32}set"
    r"|delete\s{1,32}from"
    r"|drop\s{1,32}(?:table|database)"
    r"|information_schema|sleep\s{0,8}\(|benchmark\s{0,8}\()",
    re.IGNORECASE,
)
_SQL_COMMENT_RE = re.compile(r"(?:--(?:\s|$)|/\*|\*/|#[^&\s]*)")
_SQL_TAUTOLOGY_RE = re.compile(
    # Same disjoint-class/bounded-quantifier rule as _SQL_KEYWORD_RE: adjacent optional
    # quantifiers that can both match whitespace made the original cost ~6.9 s on a
    # 32 KB whitespace run. Every ``\s*`` becomes ``\s{0,16}`` and every ``\s+`` becomes
    # ``\s{1,16}`` so the original match semantics are preserved exactly - in particular
    # zero-or-one whitespace around ``=``, which canonical ``or 1=1`` forms depend on.
    r"(?:['\"]?\s{0,16}(?:or|and)\s{1,16}['\"]?(?:\d{1,32}|[a-z]{1,32})"
    r"['\"]?\s{0,16}=\s{0,16}['\"]?(?:\d{1,32}|[a-z]{1,32})"
    r"|\bor\s{1,16}1\s{1,16}=\s{1,16}1\b)",
    re.IGNORECASE,
)
_XSS_TAG_RE = re.compile(r"<\s*/?\s*(?:script|svg|iframe|img|object|embed|style)\b", re.I)
_XSS_HANDLER_RE = re.compile(r"\bon(?:error|load|click|mouseover|focus|submit)\s*=", re.I)
_JAVASCRIPT_RE = re.compile(r"(?:javascript|vbscript)\s*:", re.I)
_TRAVERSAL_RE = re.compile(r"(?:\.\.[/\\])")
_SENSITIVE_PATH_RE = re.compile(
    r"(?:/etc/(?:passwd|shadow)|/proc/self|boot\.ini|win\.ini|windows/system32)", re.I
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+/=-]+")

# Cap on the text handed to the pattern scanners. ``percent_decode_layers`` returns
# up to five progressively decoded copies of the input, so the joined string can
# reach several hundred KB even though the raw URI/body are already truncated.
# Truncating here bounds the cost of every regex below regardless of input size.
_SCAN_LIMIT = 32768


def _body_text(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        return "&".join(f"{key}={_body_text(value)}" for key, value in body.items())
    if isinstance(body, (list, tuple)):
        return "&".join(_body_text(value) for value in body)
    return str(body)


def percent_decode_layers(value: str, max_depth: int = 4) -> tuple[str, ...]:
    """Return bounded decode layers, stopping once percent decoding stabilizes."""
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    layers = [value]
    current = value
    for _ in range(max_depth):
        decoded = urllib.parse.unquote(current, errors="replace")
        if decoded == current:
            break
        layers.append(decoded)
        current = decoded
    return tuple(layers)


def _finite_float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("HTTP feature must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError("HTTP feature must be finite")
    return converted


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return min(8.0, -sum((count / length) * math.log2(count / length) for count in counts.values()))


def extract_http_features(uri: str, body: Any = "") -> dict[str, float]:
    """Extract finite numeric features without retaining raw URI/body values."""
    raw_uri = str(uri)[:16384]
    raw_body = _body_text(body)[:32768]
    combined = f"{raw_uri}\n{raw_body}"
    layers = percent_decode_layers(combined)
    decoded = "\n".join(layers).lower()[:_SCAN_LIMIT]
    tokens = _TOKEN_RE.findall(decoded)
    query = urllib.parse.urlsplit(raw_uri).query
    features = {
        "uri_length": min(len(raw_uri), 16384),
        "body_length": min(len(raw_body), 32768),
        "query_parameter_count": min(256, query.count("&") + bool(query)),
        "percent_escape_count": min(4096, len(_PERCENT_ESCAPE_RE.findall(combined))),
        "malformed_percent_count": min(4096, len(_MALFORMED_PERCENT_RE.findall(combined))),
        "decode_depth": len(layers) - 1,
        "sql_keyword_count": min(64, len(_SQL_KEYWORD_RE.findall(decoded))),
        "sql_comment_count": min(64, len(_SQL_COMMENT_RE.findall(decoded))),
        "sql_tautology_count": min(64, len(_SQL_TAUTOLOGY_RE.findall(decoded))),
        "xss_tag_count": min(64, len(_XSS_TAG_RE.findall(decoded))),
        "xss_handler_count": min(64, len(_XSS_HANDLER_RE.findall(decoded))),
        "javascript_scheme_count": min(64, len(_JAVASCRIPT_RE.findall(decoded))),
        "traversal_count": min(64, len(_TRAVERSAL_RE.findall(decoded))),
        "sensitive_path_count": min(64, len(_SENSITIVE_PATH_RE.findall(decoded))),
        "nul_count": min(64, decoded.count("\x00")),
        "shell_metacharacter_count": min(
            256, sum(decoded.count(token) for token in (";", "|", "&&", "`", "$("))
        ),
        "max_token_length": min(4096, max((len(token) for token in tokens), default=0)),
        "character_entropy": _entropy(combined),
    }
    return {name: _finite_float(features[name]) for name in HTTP_FEATURE_NAMES}


def http_lexical_score(features: dict[str, float]) -> float:
    """Convert lexical indicators to a bounded, deterministic risk signal."""
    sql_score = min(
        0.99,
        0.70 * min(features.get("sql_tautology_count", 0.0), 1.0)
        + 0.60 * min(features.get("sql_keyword_count", 0.0), 1.0)
        + 0.25 * min(features.get("sql_comment_count", 0.0), 1.0),
    )
    xss_score = min(
        0.99,
        0.75 * min(features.get("xss_tag_count", 0.0), 1.0)
        + 0.65 * min(features.get("xss_handler_count", 0.0), 1.0)
        + 0.75 * min(features.get("javascript_scheme_count", 0.0), 1.0),
    )
    traversal_score = min(
        0.99,
        0.70 * min(features.get("traversal_count", 0.0), 1.0)
        + 0.75 * min(features.get("sensitive_path_count", 0.0), 1.0)
        + 0.30 * min(features.get("nul_count", 0.0), 1.0),
    )
    shell_score = min(0.80, 0.18 * features.get("shell_metacharacter_count", 0.0))
    score = max(sql_score, xss_score, traversal_score, shell_score)
    if score > 0 and features.get("decode_depth", 0.0) >= 2:
        score = min(0.99, score + 0.08)
    return max(0.0, min(1.0, score))
