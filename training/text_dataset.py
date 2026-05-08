"""
HDFS TraceBench → text-window builder for tier-2 transformer fine-tuning.

The tier-1 XGBoost classifier consumes per-task **bag-of-events count vectors**
(one row per task, ~2,155 columns of integer counts).  The tier-2 transformer
needs the same labelled tasks expressed as **multi-line natural-language log
windows**, because that is the surface form a real syslog stream presents.

This module reconstructs those windows from data already on disk:

  * ``eventId.json``       — list of ``[event_template_string, event_id]`` pairs
                              i.e. a vocabulary of human-readable log lines.
  * ``normal_trace.csv``    — one row per normal task; columns are event templates
                              and values are *integer counts* of how many times
                              the event occurred in that task.
  * ``failure_trace.csv``   — same schema for failure tasks.

The output is a list of ``(text, label)`` records where ``text`` is the joined
log lines for one task and ``label`` is ``0`` (normal) or ``1`` (failure).

Design choices
--------------

* **Deterministic ordering.**  Within a task we emit events in *event-id order*
  (i.e. the order columns appear in ``normal_trace.csv``).  This is reproducible
  and matches the ordering the existing tier-1 pipeline already uses.
* **One line per emission.**  An event that occurred N times is emitted N times
  (rounded down with a per-task cap to keep windows bounded).  This preserves
  intensity information that a transformer can exploit (a flood of the same
  error matters), while bounding worst-case sequence length.
* **Per-task cap.**  We cap total emissions per task at ``max_lines`` (default
  256) so the longest tasks don't blow tokenizer truncation budgets in
  unhelpful ways.  Tasks with more events keep the *highest-count* events
  preferentially so the most signal-bearing lines survive.
* **No external corpora.**  Everything we need is already in the repo.

This module is import-safe in both local Python and Kaggle environments
(no torch/transformers dependency here — the tokenizer step happens in the
trainer/notebook that consumes our output).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PREPROCESSED_DIR = Path(__file__).parent.parent / "HDFS_v3_TraceBench" / "preprocessed"

# Sizing rationale: most HDFS tasks emit < 100 distinct events but failure tasks
# can repeat the same exception hundreds of times.  256 lines * ~12 tokens/line
# ≈ 3K subword tokens which fits inside SecureBERT2.0's 1024 context after
# truncation, and inside ModernBERT-base's 8K with room to spare.
DEFAULT_MAX_LINES_PER_TASK = 256
DEFAULT_LINE_SEP = "\n"
DEFAULT_MAX_REPEAT_PER_EVENT = 16


@dataclass(frozen=True)
class TextWindow:
    """A single labelled log window for the tier-2 classifier."""

    text: str
    label: int  # 0 = normal, 1 = failure
    n_lines: int
    n_unique_events: int


def load_event_vocabulary(preprocessed_dir: Path = PREPROCESSED_DIR) -> list[str]:
    """
    Load ``eventId.json`` and return event templates in event-id order.

    The on-disk format is ``[[template, id], ...]``.  We sort by id and return
    the templates as a flat list so ``vocab[i]`` is the template for event id i.
    """
    path = preprocessed_dir / "eventId.json"
    raw = json.loads(path.read_text())
    raw_sorted = sorted(raw, key=lambda pair: int(pair[1]))
    vocab = [str(template) for template, _ in raw_sorted]
    logger.info("Loaded %d event templates from %s", len(vocab), path)
    return vocab


def _row_to_lines(
    counts: np.ndarray,
    vocab: list[str],
    max_lines: int,
    max_repeat: int,
) -> list[str]:
    """
    Convert one count vector into an ordered list of log lines.

    Strategy:
      1. Clip per-event counts to ``max_repeat`` so a single noisy event can't
         dominate the window.
      2. If total clipped emissions > ``max_lines``, keep events in *descending
         count* order (highest-signal first); break ties by event id so output
         is deterministic.
      3. Emit each surviving event ``min(count, max_repeat)`` times in event-id
         order.

    The two-pass design (select-then-emit) keeps within-task event ordering
    deterministic while still letting a flood of one event dominate naturally
    when total budget allows.
    """
    if len(vocab) != counts.shape[0]:
        raise ValueError(
            f"Vocab size {len(vocab)} does not match count vector size {counts.shape[0]}"
        )

    clipped = np.minimum(counts, max_repeat).astype(np.int64)
    total = int(clipped.sum())

    if total <= max_lines:
        kept_mask = clipped > 0
    else:
        # Need to drop events.  Rank by clipped count desc, event-id asc.
        order = np.lexsort((np.arange(len(clipped)), -clipped))
        kept_ids: list[int] = []
        running = 0
        for ev_id in order:
            c = int(clipped[ev_id])
            if c == 0:
                break  # remaining events all zero (sorted desc by count)
            if running + c > max_lines:
                # Partial keep of this event to fill the budget exactly
                remaining = max_lines - running
                if remaining > 0:
                    kept_ids.append(int(ev_id))
                    clipped[ev_id] = remaining
                    running += remaining
                break
            kept_ids.append(int(ev_id))
            running += c
        kept_mask = np.zeros_like(clipped, dtype=bool)
        kept_mask[np.array(kept_ids, dtype=np.int64)] = True

    # Emit in event-id order so windows are reproducible and the transformer
    # sees a stable structural template.
    lines: list[str] = []
    for ev_id in np.nonzero(kept_mask)[0]:
        c = int(clipped[ev_id])
        line = vocab[int(ev_id)]
        lines.extend([line] * c)
    return lines


def build_windows_from_dataframe(
    df: pd.DataFrame,
    label: int,
    vocab: list[str],
    max_lines: int = DEFAULT_MAX_LINES_PER_TASK,
    max_repeat: int = DEFAULT_MAX_REPEAT_PER_EVENT,
    line_sep: str = DEFAULT_LINE_SEP,
) -> list[TextWindow]:
    """
    Convert one trace DataFrame (rows=tasks, cols=event counts) into windows.

    The DataFrame must have the same column order as the vocabulary list
    (which the existing tier-1 loader guarantees, since ``eventId.json`` and
    the CSV headers were generated together by ``data_process.py``).
    """
    if df.shape[1] != len(vocab):
        raise ValueError(
            f"DataFrame has {df.shape[1]} columns but vocab has {len(vocab)} entries"
        )

    counts_matrix = df.to_numpy(dtype=np.int64, copy=False)
    windows: list[TextWindow] = []
    for row in counts_matrix:
        lines = _row_to_lines(row, vocab, max_lines=max_lines, max_repeat=max_repeat)
        text = line_sep.join(lines) if lines else ""
        windows.append(
            TextWindow(
                text=text,
                label=int(label),
                n_lines=len(lines),
                n_unique_events=int((row > 0).sum()),
            )
        )
    return windows


def build_windows(
    preprocessed_dir: Path = PREPROCESSED_DIR,
    sample_normal: int | None = None,
    sample_failure: int | None = None,
    max_lines: int = DEFAULT_MAX_LINES_PER_TASK,
    max_repeat: int = DEFAULT_MAX_REPEAT_PER_EVENT,
    line_sep: str = DEFAULT_LINE_SEP,
    random_state: int = 42,
) -> tuple[list[str], list[int], dict]:
    """
    End-to-end builder: load raw CSVs, render text windows, return parallel
    ``(texts, labels, stats)`` triples ready to feed into a HuggingFace
    ``Dataset``.

    Parameters mirror ``training.data_loader.load_traces`` so the tier-2
    pipeline can be sampled identically to tier-1 for fair comparison.

    Returns
    -------
    texts : list[str]
        One log window per task.
    labels : list[int]
        Parallel label list (0 = normal, 1 = failure).
    stats : dict
        Aggregate statistics useful for notebook narration / sanity checks.
    """
    vocab = load_event_vocabulary(preprocessed_dir)

    normal_path = preprocessed_dir / "normal_trace.csv"
    failure_path = preprocessed_dir / "failure_trace.csv"

    logger.info("Loading normal traces from %s", normal_path)
    df_normal = pd.read_csv(normal_path, index_col=0)
    if sample_normal is not None and sample_normal < len(df_normal):
        df_normal = df_normal.sample(n=sample_normal, random_state=random_state)
    df_normal = df_normal.astype(np.int64)

    logger.info("Loading failure traces from %s", failure_path)
    df_failure = pd.read_csv(failure_path, index_col=0)
    if sample_failure is not None and sample_failure < len(df_failure):
        df_failure = df_failure.sample(n=sample_failure, random_state=random_state)
    df_failure = df_failure.astype(np.int64)

    # Both files were written by the same preprocessing pass, so the column
    # order is identical and matches ``eventId.json``.  Defensive check anyway:
    if list(df_normal.columns) != list(df_failure.columns):
        raise ValueError("normal_trace.csv and failure_trace.csv have mismatched columns")
    if list(df_normal.columns) != vocab:
        # The tier-1 pipeline never asserted this exact equivalence because it
        # only cared about column counts.  We *need* it for tier-2 because we
        # are about to look up template strings by column index.
        logger.warning(
            "Column order in trace CSVs does not match eventId.json; "
            "reordering DataFrames to match vocab"
        )
        df_normal = df_normal.reindex(columns=vocab, fill_value=0)
        df_failure = df_failure.reindex(columns=vocab, fill_value=0)

    logger.info("Building text windows for %d normal tasks …", len(df_normal))
    normal_windows = build_windows_from_dataframe(
        df_normal,
        label=0,
        vocab=vocab,
        max_lines=max_lines,
        max_repeat=max_repeat,
        line_sep=line_sep,
    )
    logger.info("Building text windows for %d failure tasks …", len(df_failure))
    failure_windows = build_windows_from_dataframe(
        df_failure,
        label=1,
        vocab=vocab,
        max_lines=max_lines,
        max_repeat=max_repeat,
        line_sep=line_sep,
    )

    all_windows = normal_windows + failure_windows
    texts = [w.text for w in all_windows]
    labels = [w.label for w in all_windows]

    line_counts = np.array([w.n_lines for w in all_windows])
    unique_counts = np.array([w.n_unique_events for w in all_windows])
    stats = {
        "n_total": len(all_windows),
        "n_normal": len(normal_windows),
        "n_failure": len(failure_windows),
        "vocab_size": len(vocab),
        "lines_per_window": {
            "min": int(line_counts.min()) if len(line_counts) else 0,
            "p50": int(np.median(line_counts)) if len(line_counts) else 0,
            "p95": int(np.quantile(line_counts, 0.95)) if len(line_counts) else 0,
            "max": int(line_counts.max()) if len(line_counts) else 0,
            "mean": float(line_counts.mean()) if len(line_counts) else 0.0,
        },
        "unique_events_per_window": {
            "min": int(unique_counts.min()) if len(unique_counts) else 0,
            "p50": int(np.median(unique_counts)) if len(unique_counts) else 0,
            "p95": int(np.quantile(unique_counts, 0.95)) if len(unique_counts) else 0,
            "max": int(unique_counts.max()) if len(unique_counts) else 0,
        },
        "config": {
            "max_lines_per_task": max_lines,
            "max_repeat_per_event": max_repeat,
            "line_sep": repr(line_sep),
        },
    }
    logger.info("Window stats: %s", json.dumps(stats, indent=2))
    return texts, labels, stats


def _main() -> None:
    """Render a tiny preview to stdout so a human can eyeball the output."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Preview tier-2 text windows")
    parser.add_argument("--sample-normal", type=int, default=2)
    parser.add_argument("--sample-failure", type=int, default=2)
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES_PER_TASK)
    parser.add_argument("--max-repeat", type=int, default=DEFAULT_MAX_REPEAT_PER_EVENT)
    parser.add_argument("--preview-chars", type=int, default=600)
    args = parser.parse_args()

    texts, labels, stats = build_windows(
        sample_normal=args.sample_normal,
        sample_failure=args.sample_failure,
        max_lines=args.max_lines,
        max_repeat=args.max_repeat,
    )
    print(json.dumps(stats, indent=2))
    print()
    for i, (t, lab) in enumerate(zip(texts, labels)):
        kind = "NORMAL " if lab == 0 else "FAILURE"
        head = t[: args.preview_chars]
        print(f"── window[{i}] label={lab} ({kind}) chars={len(t)} ──")
        print(head)
        if len(t) > args.preview_chars:
            print(f"... [+{len(t) - args.preview_chars} more chars]")
        print()


if __name__ == "__main__":
    _main()
