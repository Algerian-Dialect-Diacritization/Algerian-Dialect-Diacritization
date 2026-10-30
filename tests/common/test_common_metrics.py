import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

from evaluation.common.data import SentenceRecord, load_jsonl
from evaluation.common.labels import apply_diacritics
from evaluation.common.paper_metrics import (
    compute_paper_metrics,
    levenshtein_distance,
    strip_diacritics,
)
from evaluation.common.projection import normalize_vocalized, project
from evaluation.common.score import append_master, load_predictions, score
from experiments.team3.llm_benchmark.build_batches import evaluation_inputs
from experiments.team3.llm_benchmark.build_batches import main as build_main
from experiments.team3.llm_benchmark.collect_outputs import main as collect_main
from experiments.team3.results.audit_numbers import NUMBER, traceable
from experiments.team3.results.build_master_table import merge


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
    # "سَارَة" vs "سَارَه": 6 NFC code points, one substitution.
    assert metrics["cer"] == pytest.approx(1 / 6)
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


def _read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_append_master_replaces_same_model_track_and_split(tmp_path: Path) -> None:
    master = tmp_path / "master.csv"
    append_master(master, {"model": "m", "track": "t1", "split": "dev", "accuracy": 0.5})
    append_master(master, {"model": "m", "track": "t1", "split": "dev", "accuracy": 0.7})
    append_master(master, {"model": "m", "track": "t1", "split": "test", "accuracy": 0.6})
    append_master(master, {"model": "m", "track": "t2", "split": "dev", "accuracy": 0.9})
    csv_rows = _read_csv(master)
    assert len(csv_rows) == 3
    rows = {(r["model"], r["track"], r["split"]): r for r in csv_rows}
    assert rows[("m", "t1", "dev")]["accuracy"] == "0.7"
    assert rows[("m", "t1", "test")]["accuracy"] == "0.6"
    assert rows[("m", "t2", "dev")]["accuracy"] == "0.9"
    assert not (tmp_path / "master.csv.tmp").exists()


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_duplicate_and_unexpected_prediction_ids_are_rejected(tmp_path: Path) -> None:
    records = _records()
    good = [{"sent_id": r.sent_id, "labels": list(r.labels)} for r in records]
    with pytest.raises(ValueError, match="duplicated"):
        load_predictions(_write_jsonl(tmp_path / "dup.jsonl", good + [good[0]]), records)
    extra = good + [{"sent_id": "999999", "labels": [0]}]
    with pytest.raises(ValueError, match="absent from gold"):
        load_predictions(_write_jsonl(tmp_path / "extra.jsonl", extra), records)


def test_duplicate_gold_ids_are_rejected(tmp_path: Path) -> None:
    row = {"sent_id": "000001", "chars": list("بت"), "labels": [1, 7], "input": "بت"}
    with pytest.raises(ValueError, match="duplicated sent_id"):
        load_jsonl(_write_jsonl(tmp_path / "gold.jsonl", [row, row]))


def test_runaway_output_is_unaligned_without_quadratic_alignment() -> None:
    record = _records()[0]
    result = project(record.chars, "ب" * 10_000)
    assert not result.skeleton_match
    assert result.unaligned_letters == 4
    assert result.labels == (0, 0, 0, 0, 0)


def _data_root(tmp_path: Path, ids: str, texts: str) -> Path:
    root = tmp_path / "data"
    (root / "test_data").mkdir(parents=True)
    (root / "test_data" / "raw_sentences_test_ids.txt").write_text(ids, encoding="utf-8")
    (root / "test_data" / "raw_sentences_test.txt").write_text(texts, encoding="utf-8")
    return root


def test_test_inputs_with_mismatched_line_counts_are_rejected(tmp_path: Path) -> None:
    root = _data_root(tmp_path, "000001\n000002\n", "بت\n")
    with pytest.raises(ValueError, match="different line counts"):
        evaluation_inputs(root, "test")
    blank = _data_root(tmp_path / "blank", "000001\n000002\n", "بت\n\nتم\n")
    with pytest.raises(ValueError):
        evaluation_inputs(blank, "test")
    ok = _data_root(tmp_path / "ok", "000001\n000002\n", "بت\nتم\n")
    assert [r["sent_id"] for r in evaluation_inputs(ok, "test")] == ["000001", "000002"]


def _run_collector(llm_dir: Path, monkeypatch) -> int:
    monkeypatch.setattr(
        sys, "argv",
        ["collect_outputs", "--llm-dir", str(llm_dir), "--condition", "zero_shot", "--split", "test"],
    )
    try:
        collect_main()
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return 0


def _llm_dir(tmp_path: Path, manifest_ids, answers: str) -> Path:
    llm = tmp_path / "llm"
    batch_dir = llm / "batches" / "zero_shot" / "test"
    batch_dir.mkdir(parents=True)
    prompt = "prompt\n"
    (batch_dir / "batch_01.prompt.md").write_text(prompt, encoding="utf-8")
    (batch_dir / "manifest.json").write_text(json.dumps({"batches": [{
        "batch": 1, "prompt_file": "batch_01.prompt.md",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "sent_ids": manifest_ids,
    }]}), encoding="utf-8")
    (llm / "inputs").mkdir()
    _write_jsonl(llm / "inputs" / "test.jsonl",
                 [{"sent_id": "000001", "input": "بت"}, {"sent_id": "000002", "input": "تم"}])
    out = llm / "claude-opus-5-5" / "zero_shot" / "test"
    out.mkdir(parents=True)
    (out / "batch_01.out.tsv").write_text(answers, encoding="utf-8")
    return llm


def test_collector_writes_predictions_only_when_complete(tmp_path: Path, monkeypatch) -> None:
    complete = "000001\tبَتْ\n000002\tتَمْ\n"
    llm = _llm_dir(tmp_path / "ok", ["000001", "000002"], complete)
    assert _run_collector(llm, monkeypatch) == 0
    predictions = llm / "claude-opus-5-5" / "predictions"
    assert (predictions / "zero_shot_test.vocalized.txt").read_text(encoding="utf-8").splitlines() == ["بَتْ", "تَمْ"]

    missing = _llm_dir(tmp_path / "missing", ["000001", "000002"], "000001\tبَتْ\n")
    assert _run_collector(missing, monkeypatch) == 1
    assert not (missing / "claude-opus-5-5" / "predictions" / "zero_shot_test.jsonl").exists()

    uncovered = _llm_dir(tmp_path / "uncovered", ["000001"], "000001\tبَتْ\n")
    assert _run_collector(uncovered, monkeypatch) == 1
    report = json.loads((uncovered / "claude-opus-5-5" / "predictions" / "zero_shot_test.collect.json").read_text())
    assert report["not_in_manifest"] == ["000002"]


def test_table_builder_ignores_test_rows_in_dev_columns(tmp_path: Path) -> None:
    scored = tmp_path / "scored.csv"
    for split, accuracy in (("dev", 0.9), ("test", 0.1)):
        append_master(scored, {
            "model": "claude-opus-5-5 (five_shot)", "track": "llm", "split": split,
            "system_type": "llm", "accuracy": accuracy, "der": 0, "der_star": 0, "der_diacritized": 0,
            "wer": 0, "wer_star": 0, "wer_aligned": 0, "cer": 0, "macro_f1": 0,
            "macro_f1_present_classes": 0, "word_accuracy": 0, "sentence_accuracy": 0,
            "skeleton_mismatch_rate": 0, "accuracy_strict": 0, "scored_letters": 1, "source": "x",
        })
    rows = {r["key"]: r for r in merge(Path("."), scored, None)}
    assert rows["llm_claude_five"]["dev_accuracy"] == 0.9


def test_kaggle_columns_follow_the_leaderboard() -> None:
    from experiments.team3.results.reported_results import all_reported

    rows = {r["key"]: r for r in all_reported(Path("."))}
    top = rows["t3_arabert_v02_09517_bilstm_crf_head"]
    assert (top["kaggle_public"], top["kaggle_private"]) == (0.95413, 0.94743)
    assert (rows["t4_souha_tcc"]["kaggle_public"], rows["t4_souha_tcc"]["kaggle_private"]) == (0.93977, 0.93344)
    assert rows["t2_canine_s"]["kaggle_private"] == 0.93235
    assert all(r["kaggle_unspecified"] is None for r in rows.values())


def test_audit_keeps_signs_and_reads_latex_ranges_as_unsigned() -> None:
    found = NUMBER.findall("gap of $-0.99$ (CI $[-1.41, +0.35]$) and 94.97--95.41, word$0.5$")
    assert found == [("-", "0.99"), ("-", "1.41"), ("+", "0.35"), ("", "94.97"), ("", "95.41"), ("", "0.5")]
    assert traceable("-", "0.25", {-0.0025})
    assert not traceable("-", "0.25", {0.25})
    assert not traceable("", "0.25", {-0.25})


def test_rebuild_is_refused_while_model_responses_exist(tmp_path: Path, monkeypatch) -> None:
    root = _data_root(tmp_path, "000001\n000002\n", "بت\nتم\n")
    out = tmp_path / "llm"
    argv = ["build_batches", "--data-root", str(root), "--out-dir", str(out),
            "--split", "test", "--condition", "zero_shot", "--num-batches", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    build_main()
    manifest = out / "batches" / "zero_shot" / "test" / "manifest.json"
    assert manifest.exists()
    response = out / "claude-opus-5-5" / "zero_shot" / "test" / "batch_01.out.tsv"
    response.parent.mkdir(parents=True)
    response.write_text("000001\tبَتْ\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        build_main()
    assert response.exists() and manifest.exists()
