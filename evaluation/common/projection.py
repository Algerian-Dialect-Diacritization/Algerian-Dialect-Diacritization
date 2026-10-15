"""Project free-form vocalized text (e.g. LLM output) onto a reference skeleton.

Mark parsing mirrors ``Data/test_data/make_submission.py``: the marks after a
letter are read as one optional Shadda plus one vowel mark, and when several
vowel marks follow one letter the last one wins.

Normalization before projection: NFC, removal of tatweel and of combining
marks outside the competition inventory, removal of punctuation, and whitespace
collapsing. If the normalized skeleton equals the reference skeleton the labels
are read directly. Otherwise the two letter sequences are aligned with a
Levenshtein backtrace; reference letters matched to an identical predicted
letter take its label, and every other reference letter is reported as
unaligned and receives label 0 (No Diacritic).
"""

import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from evaluation.common.labels import BASE_MARKS, SHADDA, apply_diacritics

VOWEL_TO_BASE = {mark: index for index, mark in enumerate(BASE_MARKS) if mark}
COMPETITION_MARKS = frozenset(VOWEL_TO_BASE) | {SHADDA}
TATWEEL = "ـ"


@dataclass(frozen=True)
class Projection:
    labels: Tuple[int, ...]
    aligned: Tuple[bool, ...]
    skeleton_match: bool
    normalized_text: str
    repaired_text: str

    @property
    def unaligned_letters(self) -> int:
        return sum(
            1 for flag in self.aligned if not flag
        )


def _units(text: str) -> List[Tuple[str, int]]:
    """Split normalized text into (base character, label) units."""
    units: List[Tuple[str, int]] = []
    shadda = False
    vowel = 0
    for char in text:
        if char in COMPETITION_MARKS:
            if not units:
                continue
            if char == SHADDA:
                shadda = True
            else:
                vowel = VOWEL_TO_BASE[char]
            base_char, _ = units[-1]
            units[-1] = (base_char, vowel + 8 * int(shadda))
            continue
        shadda = False
        vowel = 0
        units.append((char, 0))
    return [(char, 0 if char == " " else label) for char, label in units]


def normalize_vocalized(text: str) -> str:
    kept = []
    for char in unicodedata.normalize("NFC", text):
        if char == TATWEEL:
            continue
        category = unicodedata.category(char)
        if category == "Mn" and char not in COMPETITION_MARKS:
            continue
        if category.startswith("P"):
            continue
        kept.append(" " if char.isspace() else char)
    return " ".join("".join(kept).split())


def _align(reference: Sequence[str], hypothesis: Sequence[str]) -> List[int]:
    """Return, for each reference index, the matched hypothesis index or -1."""
    rows, cols = len(reference), len(hypothesis)
    cost = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        cost[i][0] = i
    for j in range(cols + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    matched = [-1] * rows
    i, j = rows, cols
    while i > 0 and j > 0:
        diagonal = cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
        if cost[i][j] == diagonal:
            if reference[i - 1] == hypothesis[j - 1]:
                matched[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return matched


def project(chars: Sequence[str], predicted_text: str) -> Projection:
    normalized = normalize_vocalized(predicted_text)
    units = _units(normalized)
    hypothesis_chars = [char for char, _ in units]
    if hypothesis_chars == list(chars):
        labels = tuple(label for _, label in units)
        aligned = tuple(True for _ in chars)
        match = True
    else:
        matched = _align(chars, hypothesis_chars)
        labels = tuple(
            0 if char == " " or index < 0 else units[index][1]
            for char, index in zip(chars, matched)
        )
        aligned = tuple(
            char == " " or index >= 0 for char, index in zip(chars, matched)
        )
        match = False
    return Projection(
        labels=labels,
        aligned=aligned,
        skeleton_match=match,
        normalized_text=normalized,
        repaired_text=apply_diacritics(chars, labels),
    )
