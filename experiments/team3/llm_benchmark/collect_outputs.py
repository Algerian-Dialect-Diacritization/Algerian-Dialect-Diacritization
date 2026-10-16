"""Merge LLM batch outputs into one prediction file and check completeness.

Reads ``batches/<condition>/<split>/manifest.json`` and the model's
``<model>/<condition>/<split>/batch_NN.out.tsv`` files (``ID<TAB>text`` per
line), then writes:

    <model>/predictions/<condition>_<split>.jsonl     {sent_id, text}
    <model>/predictions/<condition>_<split>.collect.json   completeness report
    <model>/predictions/<condition>_<split>.vocalized.txt  (test only) one
        skeleton-repaired line per input, in input order, for make_submission.py

Exits with status 1, writing only the report, when any sentence is missing,
duplicated, unexpected or absent from the manifest, or when a prompt file no
longer matches its manifest hash, so a bad batch can be re-run.
"""

import argparse
import hashlib
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
    report = {
        "batches": [], "missing": [], "duplicated": [], "unexpected": [],
        "not_in_manifest": [], "prompt_hash_mismatch": [],
    }
    manifest_ids: List[str] = []
    for batch in manifest["batches"]:
        manifest_ids.extend(batch["sent_ids"])
        prompt_path = batch_dir / batch["prompt_file"]
        if not prompt_path.exists() or hashlib.sha256(
            prompt_path.read_text(encoding="utf-8").encode()
        ).hexdigest() != batch["prompt_sha256"]:
            report["prompt_hash_mismatch"].append(batch["prompt_file"])
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
    input_ids = [row["sent_id"] for row in inputs]
    report["not_in_manifest"] = sorted(set(input_ids) - set(manifest_ids))
    report["duplicated"].extend(
        sorted({i for i in manifest_ids if manifest_ids.count(i) > 1})
    )
    report.update(sentences=len(inputs), collected=len(texts))

    prediction_dir = args.llm_dir / args.model / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    stem = "{}_{}".format(args.condition, args.split)
    problems = [key for key in ("missing", "duplicated", "unexpected", "not_in_manifest", "prompt_hash_mismatch") if report[key]]
    if problems:
        # Only the report is written, so a previous complete collection is never replaced.
        _write_atomic(prediction_dir / (stem + ".collect.json"), json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "batches"}, indent=2))
        print("collection incomplete: {}".format(", ".join(problems)), file=sys.stderr)
        sys.exit(1)

    skeleton_matches = 0
    repaired: List[str] = []
    for row in inputs:
        result = project(tuple(row["input"]), texts[row["sent_id"]])
        skeleton_matches += int(result.skeleton_match)
        repaired.append(result.repaired_text)
    _write_atomic(
        prediction_dir / (stem + ".jsonl"),
        "".join(
            json.dumps({"sent_id": row["sent_id"], "text": texts[row["sent_id"]]}, ensure_ascii=False) + "\n"
            for row in inputs
        ),
    )
    if args.split == "test":
        _write_atomic(prediction_dir / (stem + ".vocalized.txt"), "\n".join(repaired) + "\n")
    report.update(
        skeleton_match=skeleton_matches,
        skeleton_match_rate=skeleton_matches / len(inputs),
    )
    _write_atomic(prediction_dir / (stem + ".collect.json"), json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}, indent=2))


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
