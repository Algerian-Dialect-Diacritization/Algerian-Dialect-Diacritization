"""Score one prediction file with the shared paper metrics.

Prediction JSONL rows carry ``sent_id`` and either ``labels`` (one 16-class
label per gold character, spaces = 0) or ``text`` (a fully vocalized
sentence, e.g. LLM output). Text rows are projected onto the gold skeleton
with :mod:`evaluation.common.projection`; WER and CER are then computed on the
normalized text itself, so changed letters are penalized there.

Usage:
    python -m evaluation.common.score --gold dev.jsonl --pred preds.jsonl \
        --out results/<model> --model-name X --track trackN --split dev \
        [--system-type single] [--append-master MASTER_RESULTS.csv]
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.common.data import SentenceRecord, load_jsonl
from evaluation.common.paper_metrics import compute_paper_metrics
from evaluation.common.projection import project

MASTER_FIELDS = (
    "model",
    "track",
    "split",
    "system_type",
    "accuracy",
    "der",
    "der_star",
    "der_diacritized",
    "wer",
    "wer_star",
    "wer_aligned",
    "cer",
    "macro_f1",
    "macro_f1_present_classes",
    "word_accuracy",
    "sentence_accuracy",
    "skeleton_mismatch_rate",
    "accuracy_strict",
    "scored_letters",
    "source",
)


def load_predictions(
    path: Path, records: Sequence[SentenceRecord]
) -> Tuple[
    List[List[int]], Optional[List[str]], List[Tuple[bool, ...]], List[bool]
]:
    rows: Dict[str, Dict[str, Any]] = {}
    duplicated: List[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                sent_id = str(row["sent_id"])
                if sent_id in rows:
                    duplicated.append(sent_id)
                rows[sent_id] = row
    if duplicated:
        raise ValueError(
            "{} duplicated prediction ids, e.g. {}".format(len(duplicated), duplicated[:5])
        )
    missing = [record.sent_id for record in records if record.sent_id not in rows]
    if missing:
        raise ValueError(
            "{} gold sentences have no prediction, e.g. {}".format(
                len(missing), missing[:5]
            )
        )
    unexpected = sorted(set(rows) - {record.sent_id for record in records})
    if unexpected:
        raise ValueError(
            "{} predictions have ids absent from gold, e.g. {}".format(
                len(unexpected), unexpected[:5]
            )
        )
    uses_text = "text" in rows[records[0].sent_id]
    labels: List[List[int]] = []
    texts: List[str] = []
    aligned: List[Tuple[bool, ...]] = []
    matches: List[bool] = []
    for record in records:
        row = rows[record.sent_id]
        if uses_text:
            projection = project(record.chars, str(row["text"]))
            labels.append(list(projection.labels))
            texts.append(projection.normalized_text)
            aligned.append(projection.aligned)
            matches.append(projection.skeleton_match)
        else:
            labels.append([int(label) for label in row["labels"]])
            aligned.append(tuple(True for _ in record.chars))
            matches.append(True)
    return labels, (texts if uses_text else None), aligned, matches


def per_sentence_rows(
    records: Sequence[SentenceRecord],
    predictions: Sequence[Sequence[int]],
    aligned: Sequence[Tuple[bool, ...]],
    matches: Sequence[bool],
) -> List[Dict[str, Any]]:
    output = []
    for record, labels, flags, match in zip(records, predictions, aligned, matches):
        letters = [i for i, char in enumerate(record.chars) if char != " "]
        letter_errors = sum(labels[i] != record.labels[i] for i in letters)
        words, word_errors, start = 0, 0, 0
        for position in range(len(record.chars) + 1):
            if position == len(record.chars) or record.chars[position] == " ":
                if position > start:
                    words += 1
                    word_errors += int(
                        tuple(labels[start:position]) != record.labels[start:position]
                    )
                start = position + 1
        output.append(
            {
                "sent_id": record.sent_id,
                "letters": len(letters),
                "letter_errors": letter_errors,
                "words": words,
                "word_errors": word_errors,
                "unaligned_letters": sum(1 for i in letters if not flags[i]),
                "skeleton_match": match,
            }
        )
    return output


def score(
    records: Sequence[SentenceRecord],
    predictions: Sequence[Sequence[int]],
    texts: Optional[Sequence[str]],
    aligned: Sequence[Tuple[bool, ...]],
    matches: Sequence[bool],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metrics = compute_paper_metrics(records, predictions, texts)
    rows = per_sentence_rows(records, predictions, aligned, matches)
    unaligned_correct = sum(
        1
        for record, labels, flags in zip(records, predictions, aligned)
        for i, char in enumerate(record.chars)
        if char != " " and not flags[i] and labels[i] == record.labels[i]
    )
    metrics["unaligned_letters"] = sum(row["unaligned_letters"] for row in rows)
    metrics["accuracy_strict"] = (
        metrics["correct_letters"] - unaligned_correct
    ) / metrics["scored_letters"]
    metrics["skeleton_mismatch_rate"] = (
        metrics["skeleton_mismatch_count"] / metrics["sentences"]
    )
    metrics["definitions"]["accuracy_strict"] = (
        "accuracy where letters that could not be aligned to an identical "
        "predicted letter always count as errors"
    )
    return metrics, rows


def _master_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row.get("model", "")), str(row.get("track", "")), str(row.get("split", "")))


def append_master(path: Path, row: Dict[str, Any]) -> None:
    """Add or replace the row with the same (model, track, split), atomically."""
    existing: List[Dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            existing = [r for r in csv.DictReader(stream) if _master_key(r) != _master_key(row)]
    existing.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--system-type", default="single")
    parser.add_argument("--append-master", type=Path, default=None)
    args = parser.parse_args()

    records = load_jsonl(args.gold)
    predictions, texts, aligned, matches = load_predictions(args.pred, records)
    metrics, rows = score(records, predictions, texts, aligned, matches)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per_sentence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    master_row = {field: metrics.get(field, "") for field in MASTER_FIELDS}
    master_row.update(
        model=args.model_name,
        track=args.track,
        split=args.split,
        system_type=args.system_type,
        source=str(args.pred),
    )
    if args.append_master is not None:
        append_master(args.append_master, master_row)
    print(
        json.dumps(
            {field: master_row[field] for field in MASTER_FIELDS if field != "source"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
