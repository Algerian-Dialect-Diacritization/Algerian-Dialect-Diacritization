"""Check that every decimal number quoted in the §6/§7 prose is traceable.

A number is traceable when it matches (to the printed precision) a value in
MASTER_RESULTS.csv, the LLM analysis.json or an LLM metrics.json, expressed
either as a fraction or as a percentage. Untraceable numbers are listed for
manual review; integers (counts, sizes) are ignored.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Set


def walk_numbers(value) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_numbers(item)


def known_values(master: Path, json_files: Iterable[Path]) -> Set[float]:
    values: Set[float] = set()
    with master.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for cell in row.values():
                try:
                    values.add(float(cell))
                except (TypeError, ValueError):
                    pass
    for path in json_files:
        values.update(walk_numbers(json.loads(path.read_text(encoding="utf-8"))))
    return values


def traceable(number: str, values: Set[float]) -> bool:
    digits = len(number.split(".")[1])
    target = float(number)
    tolerance = 0.5 * 10 ** -digits + 1e-9
    for value in values:
        for scaled in (value, 100 * value, abs(value), 100 * abs(value)):
            if abs(scaled - target) <= tolerance:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--json", type=Path, nargs="*", default=[])
    parser.add_argument("--text", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    values = known_values(args.master, args.json)
    flagged = 0
    for path in args.text:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("%"):
                continue
            for number in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", line):
                if not traceable(number, values):
                    flagged += 1
                    print("{}:{}: {}".format(path.name, line_number, number))
    print("untraceable numbers: {}".format(flagged))


if __name__ == "__main__":
    main()
