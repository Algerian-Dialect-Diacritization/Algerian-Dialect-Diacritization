# Shared evaluator (`evaluation/common`)

One scorer for every track and for the LLM benchmark, so that numbers in the paper are computed identically. It is a track-neutral copy of `evaluation/track4/Lyes/paper_metrics.py`, extended with DER, the starred (−CE) metrics, and a text-input path for LLM outputs.

## Prediction file format

One JSON object per line, any order, one row per gold sentence:

```json
{"sent_id": "002345", "labels": [1, 0, 1, 0, 0, 1, 5, 0]}
```

`labels` has one 16-class label per element of the gold `chars` (spaces = 0). Alternatively, give the fully vocalized sentence:

```json
{"sent_id": "002345", "text": "<vocalized sentence>"}
```

Text rows are projected onto the gold skeleton (`projection.py`). Mark parsing matches `make_submission.py`. Letters that cannot be aligned count as errors in `accuracy_strict`; WER/CER use the normalized text (NFC, punctuation/tatweel/unsupported-mark removal, collapsed whitespace), so changed letters are penalized there.

## Usage

```bash
python -m evaluation.common.score \
  --gold data/dev_data/dev_Algerian-DIAC.jsonl \
  --pred my_dev_predictions.jsonl \
  --out results/my_model --model-name "My model" --track track1 --split dev \
  --append-master results/MASTER_RESULTS_scored.csv
```

This writes `metrics.json` (all metrics, per-class P/R/F1, confusion matrix) and `per_sentence.csv` (for bootstrap tests and error analysis).

## Metrics

| Metric | Definition |
|---|---|
| accuracy (= micro-F1), der | over all non-space letters, gold "no diacritic" letters included; DER = 1 − accuracy |
| der_star, wer_star | final letter of each multi-letter word excluded (−CE); one-letter words excluded (same convention as Track 1/3's `utils/evaluation_metrics.py`, branch `Track1/Thabet/P2-Strategy-A`) |
| wer, cer | Levenshtein over vocalized words / NFC code points (spaces included) |
| wer_aligned | 1 − word accuracy (Track 1/3 WER definition) |
| der_diacritized | DER over letters whose gold label is not "no diacritic" |
| macro_f1, macro_f1_present_classes | over all 16 classes / over classes present in gold |

Tests: `python -m pytest tests/common -q`.
