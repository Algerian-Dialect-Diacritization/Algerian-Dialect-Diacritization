"""Torch-free loading of labelled sentence records for evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from evaluation.common.labels import NUM_LABELS


@dataclass(frozen=True)
class SentenceRecord:
    sent_id: str
    chars: Tuple[str, ...]
    labels: Optional[Tuple[int, ...]]
    input_text: str
    target_text: Optional[str] = None


def record_from_mapping(raw: Mapping[str, Any]) -> SentenceRecord:
    """Validate one competition JSONL row and return a record."""
    missing = sorted({"sent_id", "chars", "labels", "input"}.difference(raw))
    if missing:
        raise ValueError("record is missing keys {}".format(", ".join(missing)))
    sent_id = str(raw["sent_id"])
    chars = tuple(str(char) for char in raw["chars"])
    labels = tuple(int(label) for label in raw["labels"])
    input_text = str(raw["input"])
    if not chars or any(len(char) != 1 for char in chars):
        raise ValueError("{}: chars must be nonempty single code points".format(sent_id))
    if len(chars) != len(labels):
        raise ValueError("{}: chars/labels length mismatch".format(sent_id))
    if "".join(chars) != input_text:
        raise ValueError("{}: input does not equal joined chars".format(sent_id))
    if any(label < 0 or label >= NUM_LABELS for label in labels):
        raise ValueError("{}: label outside [0, 15]".format(sent_id))
    if any(char == " " and label != 0 for char, label in zip(chars, labels)):
        raise ValueError("{}: non-zero space label".format(sent_id))
    target = raw.get("target")
    return SentenceRecord(
        sent_id=sent_id,
        chars=chars,
        labels=labels,
        input_text=input_text,
        target_text=None if target is None else str(target),
    )


def load_jsonl(path: Path) -> List[SentenceRecord]:
    records: List[SentenceRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(record_from_mapping(json.loads(line)))
    if not records:
        raise ValueError("{} contains no records".format(path))
    return records
