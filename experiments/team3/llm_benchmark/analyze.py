"""§7 error analysis: lexicon buckets, dialect indicators, paired bootstrap.

Every system is given as NAME=PATH where PATH is a prediction JSONL with
``labels`` or ``text`` rows (text rows are projected as in the scorer).

Outputs, under ``--out`` (keep outside the public repo; it quotes sentences):

    analysis.json            numbers used in §7
    example_candidates.md    sentences for the qualitative analysis
"""

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from evaluation.common.data import SentenceRecord, load_jsonl
from evaluation.common.labels import LABEL_NAMES, apply_diacritics
from evaluation.common.score import load_predictions

TANWEEN = {2, 4, 6, 10, 12, 14}
SHORT_VOWELS = {1, 3, 5, 9, 11, 13}
SUKUN = {7, 15}


def _rate(numerator: int, denominator: int):
    """Ratio, or None when the group is empty (reported as null in analysis.json)."""
    return numerator / denominator if denominator else None


def words_of(record: SentenceRecord) -> List[Tuple[int, int]]:
    spans, start = [], 0
    for position in range(len(record.chars) + 1):
        if position == len(record.chars) or record.chars[position] == " ":
            if position > start:
                spans.append((start, position))
            start = position + 1
    return spans


def train_lexicon(records: Sequence[SentenceRecord]) -> Dict[str, collections.Counter]:
    lexicon: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in records:
        for start, stop in words_of(record):
            lexicon["".join(record.chars[start:stop])][record.labels[start:stop]] += 1
    return lexicon


def bucket(word: str, lexicon: Dict[str, collections.Counter]) -> str:
    if word not in lexicon:
        return "oov"
    return "seen_ambiguous" if len(lexicon[word]) > 1 else "seen_unambiguous"


def position_rates(records, predictions) -> Dict[str, float]:
    """Dialect-sensitive label rates at word-initial and word-final letters."""
    counts = collections.Counter()
    for record, labels in zip(records, predictions):
        for start, stop in words_of(record):
            if stop - start < 2:
                continue
            counts["initial"] += 1
            counts["initial_sukun"] += int(labels[start] in SUKUN)
            counts["final"] += 1
            counts["final_short_vowel"] += int(labels[stop - 1] in SHORT_VOWELS)
            counts["final_tanween"] += int(labels[stop - 1] in TANWEEN)
            counts["final_sukun"] += int(labels[stop - 1] in SUKUN)
            counts["final_none"] += int(labels[stop - 1] == 0)
    return {
        "word_initial_sukun_rate": _rate(counts["initial_sukun"], counts["initial"]),
        "word_final_short_vowel_rate": _rate(counts["final_short_vowel"], counts["final"]),
        "word_final_tanween_rate": _rate(counts["final_tanween"], counts["final"]),
        "word_final_sukun_rate": _rate(counts["final_sukun"], counts["final"]),
        "word_final_no_diacritic_rate": _rate(counts["final_none"], counts["final"]),
        "multi_letter_words": counts["initial"],
    }


def bucket_wer(records, predictions, lexicon) -> Dict[str, Dict[str, float]]:
    totals = collections.Counter()
    errors = collections.Counter()
    for record, labels in zip(records, predictions):
        for start, stop in words_of(record):
            name = bucket("".join(record.chars[start:stop]), lexicon)
            totals[name] += 1
            errors[name] += int(tuple(labels[start:stop]) != record.labels[start:stop])
    return {
        name: {"words": totals[name], "wer": _rate(errors[name], totals[name])}
        for name in ("seen_unambiguous", "seen_ambiguous", "oov")
    }


def sentence_correct_letters(records, predictions) -> List[Tuple[int, int]]:
    rows = []
    for record, labels in zip(records, predictions):
        letters = [i for i, char in enumerate(record.chars) if char != " "]
        rows.append((sum(labels[i] == record.labels[i] for i in letters), len(letters)))
    return rows


def paired_bootstrap(a, b, samples: int, seed: int) -> Dict[str, float]:
    """Sentence-level paired bootstrap of the letter-accuracy difference a - b.

    Significance is read from the 95% percentile interval (does it exclude 0?).
    ``bootstrap_prob_a_better`` is the share of resamples where a beats b; it is
    a bootstrap probability, not a p-value.
    """
    rng = random.Random(seed)
    n = len(a)
    observed = sum(x for x, _ in a) / sum(t for _, t in a) - sum(x for x, _ in b) / sum(t for _, t in b)
    wins = 0
    deltas = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        total = sum(a[i][1] for i in idx)
        delta = (sum(a[i][0] for i in idx) - sum(b[i][0] for i in idx)) / total
        deltas.append(delta)
        wins += int(delta > 0)
    deltas.sort()
    return {
        "delta_accuracy": observed,
        "ci95_low": deltas[int(0.025 * samples)],
        "ci95_high": deltas[int(0.975 * samples) - 1],
        "bootstrap_prob_a_better": wins / samples,
        "samples": samples,
    }


def top_confusions(records, predictions, k=10):
    pairs = collections.Counter()
    for record, labels in zip(records, predictions):
        for char, gold, pred in zip(record.chars, record.labels, labels):
            if char != " " and gold != pred:
                pairs[(LABEL_NAMES[gold], LABEL_NAMES[pred])] += 1
    return [{"gold": g, "pred": p, "count": c} for (g, p), c in pairs.most_common(k)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--system", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--compare", action="append", default=[], help="NAME_A:NAME_B")
    parser.add_argument("--examples-for", default=None, help="system NAME for example mining")
    parser.add_argument("--reference-system", default=None, help="fine-tuned NAME shown next to examples")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    records = load_jsonl(args.gold)
    lexicon = train_lexicon(load_jsonl(args.train))
    systems = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        systems[name] = load_predictions(Path(path), records)[0]

    analysis = {
        "gold": {
            "position_rates": position_rates(records, [r.labels for r in records]),
        },
        "systems": {},
        "comparisons": {},
    }
    for name, predictions in systems.items():
        analysis["systems"][name] = {
            "position_rates": position_rates(records, predictions),
            "bucket_wer": bucket_wer(records, predictions, lexicon),
            "top_confusions": top_confusions(records, predictions),
        }
    for spec in args.compare:
        first, second = spec.split(":")
        analysis["comparisons"][spec] = paired_bootstrap(
            sentence_correct_letters(records, systems[first]),
            sentence_correct_letters(records, systems[second]),
            args.bootstrap,
            seed=42,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.examples_for:
        mined = systems[args.examples_for]
        reference = systems.get(args.reference_system) if args.reference_system else None
        lines = ["# Example candidates ({})\n".format(args.examples_for)]
        categories = collections.defaultdict(list)
        for index, (record, labels) in enumerate(zip(records, mined)):
            for start, stop in words_of(record):
                gold_word = record.labels[start:stop]
                pred_word = tuple(labels[start:stop])
                if gold_word == pred_word or stop - start < 2:
                    continue
                word = "".join(record.chars[start:stop])
                if pred_word[0] not in SUKUN and gold_word[0] in SUKUN:
                    categories["initial_vowel_instead_of_sukun"].append(index)
                if pred_word[-1] in SHORT_VOWELS | TANWEEN and gold_word[-1] not in SHORT_VOWELS | TANWEEN:
                    categories["msa_style_final_vowel"].append(index)
                if bucket(word, lexicon) == "oov":
                    categories["oov_word"].append(index)
        for name, indices in categories.items():
            lines.append("\n## {} ({} word errors)\n".format(name, len(indices)))
            for index in sorted(set(indices))[:12]:
                record = records[index]
                lines.append("- `{}`".format(record.sent_id))
                lines.append("  - gold: {}".format(apply_diacritics(record.chars, record.labels)))
                lines.append("  - {}: {}".format(args.examples_for, apply_diacritics(record.chars, mined[index])))
                if reference is not None:
                    lines.append("  - {}: {}".format(args.reference_system, apply_diacritics(record.chars, reference[index])))
        (args.out / "example_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in analysis.items() if k != "systems"}, indent=2))


if __name__ == "__main__":
    main()
