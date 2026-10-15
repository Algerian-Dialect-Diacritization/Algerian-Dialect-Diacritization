import json
from pathlib import Path

import pytest

from evaluation.common.data import SentenceRecord
from evaluation.common.labels import apply_diacritics
from evaluation.common.paper_metrics import (
    compute_paper_metrics,
    levenshtein_distance,
    strip_diacritics,
)
from evaluation.common.projection import normalize_vocalized, project
from evaluation.common.score import append_master, load_predictions, score


def _records() -> list:
    return [
        SentenceRecord(
            sent_id="000001",
            chars=tuple("بت تم"),
            labels=(1, 7, 0, 9, 2),
            input_text="بت تم",
        ),
        SentenceRecord(
            sent_id="000002",
            chars=tuple("ب"),
            labels=(3,),
            input_text="ب",
        ),
    ]


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("", "", 0),
        ("kitten", "sitting", 3),
        (tuple("سارة"), tuple("ساره"), 1),
    ],
)
def test_levenshtein_distance_is_exact(reference, hypothesis, expected) -> None:
    assert levenshtein_distance(reference, hypothesis) == expected


def test_perfect_predictions_have_zero_error_everywhere() -> None:
    records = _records()
    metrics = compute_paper_metrics(records, [list(r.labels) for r in records])
    for key in ("der", "der_star", "wer", "wer_star", "wer_aligned", "cer", "der_diacritized"):
        assert metrics[key] == 0.0, key
    assert metrics["accuracy"] == 1.0
    # Two multi-letter words with one non-final letter each; "ب" is excluded.
    assert metrics["starred_letter_total"] == 2
    assert metrics["starred_word_total"] == 2


def test_final_letter_errors_are_excluded_from_starred_metrics() -> None:
    records = _records()
    predictions = [[1, 0, 0, 9, 0], [3]]  # only word-final letters wrong
    metrics = compute_paper_metrics(records, predictions)
    assert metrics["der"] == pytest.approx(2 / 5)
    assert metrics["wer_aligned"] == pytest.approx(2 / 3)
    assert metrics["der_star"] == 0.0
    assert metrics["wer_star"] == 0.0


def test_non_final_errors_count_in_starred_metrics() -> None:
    records = _records()
    predictions = [[5, 7, 0, 9, 2], [3]]
    metrics = compute_paper_metrics(records, predictions)
    assert metrics["der_star"] == pytest.approx(1 / 2)
    assert metrics["wer_star"] == pytest.approx(1 / 2)
    # gold-diacritized letters: all 5 letters carry a mark, one is wrong
    assert metrics["der_diacritized"] == pytest.approx(1 / 5)


def test_projection_round_trips_gold_text() -> None:
    record = _records()[0]
    text = apply_diacritics(record.chars, record.labels)
    result = project(record.chars, text)
    assert result.skeleton_match
    assert result.labels == record.labels
    assert result.repaired_text == text


def test_projection_ignores_punctuation_tatweel_and_extra_spaces() -> None:
    record = _records()[0]
    text = apply_diacritics(record.chars, record.labels)
    noisy = "  " + text[:2] + "ـ" + text[2:] + " ؟ "
    assert strip_diacritics(normalize_vocalized(noisy)) == record.input_text
    result = project(record.chars, noisy)
    assert result.skeleton_match
    assert result.labels == record.labels


def test_projection_last_vowel_wins_like_make_submission() -> None:
    result = project(tuple("ب"), "بَِ")
    assert result.labels == (5,)
    result = project(tuple("ب"), "بِّ")
    assert result.labels == (13,)


def test_changed_letter_is_unaligned_and_strict_accuracy_drops() -> None:
    record = SentenceRecord(
        sent_id="000003", chars=tuple("سارة"), labels=(1, 0, 1, 0), input_text="سارة"
    )
    result = project(record.chars, "سَارَه")
    assert not result.skeleton_match
    assert result.aligned == (True, True, True, False)
    assert result.labels == (1, 0, 1, 0)
    metrics, rows = score([record], [list(result.labels)], [result.normalized_text], [result.aligned], [False])
    assert metrics["accuracy"] == 1.0  # label 0 on the unaligned letter happens to match gold
    assert metrics["accuracy_strict"] == pytest.approx(3 / 4)
    assert metrics["skeleton_mismatch_count"] == 1
    assert metrics["cer"] > 0.0
    assert rows[0]["unaligned_letters"] == 1


def test_text_and_label_prediction_files_score_identically(tmp_path: Path) -> None:
    records = _records()
    label_path = tmp_path / "labels.jsonl"
    text_path = tmp_path / "text.jsonl"
    predictions = [[1, 0, 0, 9, 2], [5]]
    with label_path.open("w", encoding="utf-8") as stream:
        for record, labels in zip(records, predictions):
            stream.write(json.dumps({"sent_id": record.sent_id, "labels": labels}) + "\n")
    with text_path.open("w", encoding="utf-8") as stream:
        for record, labels in reversed(list(zip(records, predictions))):
            text = apply_diacritics(record.chars, labels)
            stream.write(json.dumps({"sent_id": record.sent_id, "text": text}, ensure_ascii=False) + "\n")
    from_labels, _ = score(records, *load_predictions(label_path, records))
    from_text, _ = score(records, *load_predictions(text_path, records))
    for key in ("accuracy", "der_star", "wer", "wer_star", "cer"):
        assert from_labels[key] == pytest.approx(from_text[key]), key


def test_append_master_replaces_same_model_and_split(tmp_path: Path) -> None:
    master = tmp_path / "master.csv"
    append_master(master, {"model": "m", "split": "dev", "accuracy": 0.5})
    append_master(master, {"model": "m", "split": "dev", "accuracy": 0.7})
    append_master(master, {"model": "m", "split": "test", "accuracy": 0.6})
    lines = master.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert ",0.7," in lines[1]
