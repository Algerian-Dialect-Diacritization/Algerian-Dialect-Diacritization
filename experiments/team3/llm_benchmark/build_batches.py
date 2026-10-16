"""Freeze the §7 LLM benchmark protocol and write self-contained batch prompts.

Only the undiacritized ``input`` field of each evaluation sentence is ever
written into a prompt. Gold targets of the evaluation split are not read here.
Few-shot examples come from the training split only.

Outputs (all under ``--out-dir``, which should live OUTSIDE the public repo
because the files contain corpus sentences):

    inputs/<split>.jsonl                        {sent_id, input}
    fewshot.json                                fixed few-shot examples (train)
    batches/<condition>/<split>/batch_NN.prompt.md
    batches/<condition>/<split>/manifest.json   ids per batch + prompt sha256
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

INSTRUCTIONS = """\
You are given sentences in Algerian Arabic (Darja) written in Arabic script without diacritics.
Add the diacritics (tashkeel) to every sentence.

Rules:
- Use only these marks: fatha, damma, kasra, sukun, fathatan, dammatan, kasratan, shadda, or shadda combined with one of the other marks.
- Put at most one mark (optionally with shadda) after each letter.
- Do not add, remove, reorder or change any letter, and do not change the spacing.
- Vocalize each sentence as it is pronounced in Algerian Arabic, not as Modern Standard Arabic.
- Output exactly one line per input line, in the format: ID<TAB>diacritized sentence
- Output nothing else: no commentary, no blank lines, no code fences.
"""

FEWSHOT_HEADER = "Examples of correctly diacritized Algerian Arabic sentences:\n"


def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluation_inputs(data_root: Path, split: str) -> List[Dict[str, str]]:
    if split == "dev":
        rows = load_jsonl(data_root / "dev_data" / "dev_Algerian-DIAC.jsonl")
        return [{"sent_id": row["sent_id"], "input": row["input"]} for row in rows]
    texts = (data_root / "test_data" / "raw_sentences_test.txt").read_text(
        encoding="utf-8"
    ).split("\n")
    ids = (data_root / "test_data" / "raw_sentences_test_ids.txt").read_text(
        encoding="utf-8"
    ).split("\n")
    while ids and not ids[-1].strip():
        ids.pop()
    while texts and not texts[-1].strip():
        texts.pop()
    if len(ids) != len(texts):
        raise ValueError(
            "test ids ({}) and sentences ({}) have different line counts".format(len(ids), len(texts))
        )
    if any(not i.strip() for i in ids):
        raise ValueError("test ids file contains blank lines")
    return [{"sent_id": i.strip(), "input": t.strip()} for i, t in zip(ids, texts)]


def select_fewshot(data_root: Path, k: int, seed: int) -> List[Dict[str, str]]:
    """One training sentence per length quantile, chosen with a fixed seed."""
    rows = load_jsonl(data_root / "train_data" / "train_Algerian-DIAC.jsonl")
    rows = [row for row in rows if "-" not in row["sent_id"]]
    rows.sort(key=lambda row: (len(row["input"]), row["sent_id"]))
    rng = random.Random(seed)
    chosen = []
    for bucket in range(k):
        start = bucket * len(rows) // k
        stop = (bucket + 1) * len(rows) // k
        row = rows[rng.randrange(start, stop)]
        chosen.append(
            {"sent_id": row["sent_id"], "input": row["input"], "target": row["target"]}
        )
    return chosen


def render_prompt(
    batch: Sequence[Dict[str, str]], fewshot: Sequence[Dict[str, str]]
) -> str:
    parts = [INSTRUCTIONS]
    if fewshot:
        parts.append(FEWSHOT_HEADER)
        parts.append(
            "\n".join(
                "Input: {}\nOutput: {}".format(ex["input"], ex["target"])
                for ex in fewshot
            )
            + "\n"
        )
    parts.append("Sentences (ID<TAB>sentence):\n")
    parts.append("\n".join("{}\t{}".format(r["sent_id"], r["input"]) for r in batch))
    return "\n".join(parts) + "\n"


def split_batches(rows: Sequence[Dict], num_batches: int) -> List[List[Dict]]:
    size, remainder = divmod(len(rows), num_batches)
    batches, start = [], 0
    for index in range(num_batches):
        stop = start + size + (1 if index < remainder else 0)
        batches.append(list(rows[start:stop]))
        start = stop
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--condition", choices=("zero_shot", "five_shot"), required=True
    )
    parser.add_argument("--num-batches", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = evaluation_inputs(args.data_root, args.split)
    if not 1 <= args.num_batches <= len(rows):
        parser.error("--num-batches must be between 1 and {}".format(len(rows)))
    inputs_path = args.out_dir / "inputs" / "{}.jsonl".format(args.split)
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    with inputs_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    fewshot: List[Dict[str, str]] = []
    if args.condition == "five_shot":
        fewshot = select_fewshot(args.data_root, 5, args.seed)
        (args.out_dir / "fewshot.json").write_text(
            json.dumps(fewshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    batch_dir = args.out_dir / "batches" / args.condition / args.split
    batch_dir.mkdir(parents=True, exist_ok=True)
    # Drop any previous manifest first: an interrupted rebuild then leaves no
    # manifest at all instead of an old manifest describing new prompts.
    (batch_dir / "manifest.json").unlink(missing_ok=True)
    for stale in batch_dir.glob("batch_*.prompt.md"):
        stale.unlink()
    manifest = {
        "split": args.split,
        "condition": args.condition,
        "seed": args.seed,
        "instructions_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        "fewshot_ids": [ex["sent_id"] for ex in fewshot],
        "batches": [],
    }
    for index, batch in enumerate(split_batches(rows, args.num_batches), start=1):
        prompt = render_prompt(batch, fewshot)
        path = batch_dir / "batch_{:02d}.prompt.md".format(index)
        path.write_text(prompt, encoding="utf-8")
        manifest["batches"].append(
            {
                "batch": index,
                "prompt_file": path.name,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "sent_ids": [row["sent_id"] for row in batch],
            }
        )
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "{} {}: {} sentences in {} batches -> {}".format(
            args.split, args.condition, len(rows), args.num_batches, batch_dir
        )
    )


if __name__ == "__main__":
    main()
