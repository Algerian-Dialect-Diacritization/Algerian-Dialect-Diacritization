"""Merge LLM batch outputs into one prediction file and check completeness.

Reads ``batches/<condition>/<split>/manifest.json`` and the model's
``<model>/<condition>/<split>/batch_NN.out.tsv`` files (``ID<TAB>text`` per
line), then writes:

    <model>/predictions/<condition>_<split>.jsonl     {sent_id, text}
    <model>/predictions/<condition>_<split>.collect.json   completeness report
    <model>/predictions/<condition>_<split>.vocalized.txt  (test only) one
        skeleton-repaired line per input, in input order, for make_submission.py

Exits with status 1 when any sentence is missing so the batch can be re-run.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from evaluation.common.projection import project

LINE = re.compile(r"^\s*([0-9]{6}(?:-[0-9]+)?)\s*[\t ]\s*(.+?)\s*$")


def parse_batch(path: Path) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    if not path.exists():
        return found
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith("```"):
            continue
        match = LINE.match(raw)
        if match:
            found.setdefault(match.group(1), []).append(match.group(2))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-dir", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    args = parser.parse_args()

    batch_dir = args.llm_dir / "batches" / args.condition / args.split
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    inputs = [
        json.loads(line)
        for line in (args.llm_dir / "inputs" / "{}.jsonl".format(args.split))
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    output_dir = args.llm_dir / args.model / args.condition / args.split
    texts: Dict[str, str] = {}
    report = {"batches": [], "missing": [], "duplicated": [], "unexpected": []}
    for batch in manifest["batches"]:
        found = parse_batch(output_dir / batch["prompt_file"].replace(".prompt.md", ".out.tsv"))
        expected = set(batch["sent_ids"])
        missing = [i for i in batch["sent_ids"] if i not in found]
        report["batches"].append(
            {"batch": batch["batch"], "expected": len(expected), "missing": len(missing)}
        )
        report["missing"].extend(missing)
        report["unexpected"].extend(sorted(set(found) - expected))
        for sent_id, candidates in found.items():
            if sent_id in expected:
                if len(candidates) > 1:
                    report["duplicated"].append(sent_id)
                texts[sent_id] = candidates[0]

    prediction_dir = args.llm_dir / args.model / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    stem = "{}_{}".format(args.condition, args.split)
    skeleton_matches = 0
    with (prediction_dir / (stem + ".jsonl")).open("w", encoding="utf-8") as stream:
        for row in inputs:
            if row["sent_id"] in texts:
                stream.write(
                    json.dumps({"sent_id": row["sent_id"], "text": texts[row["sent_id"]]}, ensure_ascii=False)
                    + "\n"
                )
    repaired: List[str] = []
    for row in inputs:
        result = project(tuple(row["input"]), texts.get(row["sent_id"], ""))
        skeleton_matches += int(result.skeleton_match)
        repaired.append(result.repaired_text)
    if args.split == "test":
        (prediction_dir / (stem + ".vocalized.txt")).write_text(
            "\n".join(repaired) + "\n", encoding="utf-8"
        )
    report.update(
        sentences=len(inputs),
        collected=len(texts),
        skeleton_match=skeleton_matches,
        skeleton_match_rate=skeleton_matches / len(inputs),
    )
    (prediction_dir / (stem + ".collect.json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}, indent=2))
    if report["missing"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
