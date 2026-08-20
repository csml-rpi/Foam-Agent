#!/usr/bin/env python3
"""Scan a Foundation tutorial tree through Foam-Agent's import preflight.

The script deliberately calls :func:`import_case` only: every candidate is
copied into an isolated temporary output, validated, and represented by a
manifest, but no solver is launched.  It is useful in Docker/CI to measure
which upstream tutorial shapes are accepted by the intentionally restricted
case-import policy.

Example::

    source /opt/openfoam10/etc/bashrc
    python scripts/scan_case_import_matrix.py --tutorial-root "$FOAM_TUTORIALS"
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from services.case_import import CaseImportError, import_case  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight all OpenFOAM tutorials through safe case-import mode."
    )
    parser.add_argument(
        "--tutorial-root",
        type=Path,
        default=os.environ.get("FOAM_TUTORIALS"),
        help="Tutorial root (defaults to $FOAM_TUTORIALS).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional positive cap for a quick smoke scan; 0 scans every case.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the machine-readable report.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print aggregate counts only (the optional JSON report remains complete).",
    )
    return parser.parse_args()


def _tutorial_root(args: argparse.Namespace) -> Path | None:
    if args.tutorial_root is None:
        print("--tutorial-root is required when FOAM_TUTORIALS is not set.", file=sys.stderr)
        return None
    if args.max_cases < 0:
        print("--max-cases must be zero or positive.", file=sys.stderr)
        return None
    tutorial_root = args.tutorial_root.expanduser().resolve()
    if not tutorial_root.is_dir():
        print(f"Tutorial root is not a directory: {tutorial_root}", file=sys.stderr)
        return None
    return tutorial_root


def _tutorial_cases(tutorial_root: Path, max_cases: int) -> list[Path]:
    cases = sorted(
        control.parent.parent
        for control in tutorial_root.rglob("controlDict")
        if control.parent.name == "system" and control.is_file()
    )
    return cases[:max_cases] if max_cases else cases


def _scan_cases(tutorial_root: Path, cases: list[Path]) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    blocking_reasons: Counter[str] = Counter()
    rejected: list[dict[str, str]] = []
    supported: list[str] = []
    blocked: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="foamagent-import-matrix-") as scratch:
        scratch_root = Path(scratch)
        for index, case in enumerate(cases, start=1):
            relative = case.relative_to(tutorial_root).as_posix()
            try:
                manifest = import_case(case, scratch_root / f"case-{index}")
            except CaseImportError as exc:
                outcomes["rejected"] += 1
                rejected.append({"case": relative, "error": str(exc)})
                continue
            if manifest.supported:
                outcomes["supported"] += 1
                supported.append(relative)
                continue
            outcomes["blocked"] += 1
            for issue in manifest.blocking_issues:
                blocking_reasons[issue] += 1
            blocked.append({"case": relative, "issues": manifest.blocking_issues})
    return {
        "tutorial_root": str(tutorial_root),
        "cases_scanned": len(cases),
        "outcomes": dict(outcomes),
        "blocking_reason_counts": dict(blocking_reasons.most_common()),
        "supported": supported,
        "blocked": blocked,
        "rejected": rejected,
    }


def _write_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    visible_keys = (
        "tutorial_root",
        "cases_scanned",
        "outcomes",
        "blocking_reason_counts",
    )
    printable_report = {key: report[key] for key in visible_keys} if args.summary_only else report
    print(json.dumps(printable_report, indent=2, ensure_ascii=False))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    args = _parse_args()
    tutorial_root = _tutorial_root(args)
    if tutorial_root is None:
        return 2

    report = _scan_cases(tutorial_root, _tutorial_cases(tutorial_root, args.max_cases))
    _write_report(report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
