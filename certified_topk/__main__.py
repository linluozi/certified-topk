"""Command line entry point: ``python -m certified_topk [--capacity N] [--k K] [FILE]``.

Reads one item per line (blank lines ignored) or, with ``--json``, a JSON array of
items / JSON-lines records with an ``item`` field.  Exit code 0 when the top-k is
certified, 1 when it is not, 2 on usage or input errors.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import CertifiedTopK


def _read_items(path: str, json_mode: bool):
    fh = open(path, "r", encoding="utf-8") if path != "-" else sys.stdin
    try:
        if json_mode:
            payload = json.load(fh)
            if isinstance(payload, list):
                yield from payload
            else:
                yield from (row["item"] for row in payload)
        else:
            for line in fh:
                item = line.strip()
                if item:
                    yield item
    finally:
        if path != "-":
            fh.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="certified_topk")
    ap.add_argument("file", nargs="?", default="-", help="input file, - for stdin")
    ap.add_argument("--capacity", type=int, default=1024)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="input is JSON")
    args = ap.parse_args(argv)
    try:
        sketch = CertifiedTopK(args.capacity)
        for item in _read_items(args.file, args.json):
            sketch.add(item)
        result = sketch.query(args.k)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"certified_topk: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
