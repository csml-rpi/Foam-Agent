#!/usr/bin/env python3
"""
Extract a v10 case from the RAG database, run ESI translation, and optionally execute OpenFOAM.

Usage examples:
  # Translate an existing case directory
  python scripts/verify_esi_translation.py /tmp/cavity_v10

  # Extract preset, translate, run blockMesh + solver (OpenFOAM must be in PATH)
  python scripts/verify_esi_translation.py --preset cavity -o /tmp/cavity_v10 --run

  # Extract only
  python scripts/verify_esi_translation.py --preset pitzDaily -o /tmp/pitz_v10 --extract-only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from translation.esi_translator import ESITranslator  # noqa: E402
from extract_v10_case import _PRESETS, extract_case  # noqa: E402


class _EsiConfig:
  """Minimal config object for ESITranslator."""

  openfoam_fork = "esi"


def _read_application(case_dir: Path) -> str:
  control = case_dir / "system" / "controlDict"
  if not control.is_file():
    raise FileNotFoundError(f"Missing {control}")
  text = control.read_text(encoding="utf-8", errors="ignore")
  match = re.search(r"application\s+(\w+)\s*;", text)
  if not match:
    raise ValueError(f"Could not parse application from {control}")
  return match.group(1)


def _run_openfoam(case_dir: Path, application: str, timeout: int) -> None:
  case_dir = case_dir.resolve()
  print(f"\n[run] blockMesh in {case_dir}")
  subprocess.run(
    ["blockMesh"],
    cwd=case_dir,
    check=True,
    timeout=timeout,
  )
  print(f"[run] {application}")
  subprocess.run(
    [application],
    cwd=case_dir,
    check=True,
    timeout=timeout,
  )


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Verify ESI translation on database-extracted v10 cases."
  )
  parser.add_argument(
    "case_dir",
    nargs="?",
    type=Path,
    help="Existing case directory to translate (skip extract)",
  )
  parser.add_argument("-o", "--output", type=Path, help="Case directory (for extract)")
  parser.add_argument("--preset", choices=sorted(_PRESETS), help="Built-in extract preset")
  parser.add_argument("--case-name", help="Custom extract: case name")
  parser.add_argument("--solver", help="Custom extract: solver filter")
  parser.add_argument("--domain", help="Custom extract: domain filter")
  parser.add_argument("--category", help="Custom extract: category filter")
  parser.add_argument("--db", type=Path, default=ROOT / "database" / "raw" / "openfoam_tutorials_details.txt")
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument(
    "--extract-only",
    action="store_true",
    help="Only extract from database; do not translate",
  )
  parser.add_argument(
    "--skip-translate",
    action="store_true",
    help="Skip translation (case_dir must already be ESI-ready)",
  )
  parser.add_argument(
    "--run",
    action="store_true",
    help="Run blockMesh and the case application after translation",
  )
  parser.add_argument(
    "--timeout",
    type=int,
    default=600,
    help="Timeout per OpenFOAM command in seconds (default: 600)",
  )
  args = parser.parse_args()

  case_dir: Path | None = args.case_dir

  if args.preset or args.case_name:
    if not args.output:
      parser.error("-o/--output is required when using --preset or --case-name")
    if args.preset:
      preset = _PRESETS[args.preset]
      case_name = preset["case_name"]
      solver = preset.get("solver")
      domain = preset.get("domain")
      category = preset.get("category")
    else:
      case_name = args.case_name
      solver = args.solver
      domain = args.domain
      category = args.category

    print(f"[extract] -> {args.output}")
    try:
      extract_case(
        output_dir=args.output,
        case_name=case_name,
        db_path=args.db,
        solver=solver,
        domain=domain,
        category=category,
        overwrite=args.overwrite,
      )
    except (FileNotFoundError, LookupError, FileExistsError) as exc:
      print(f"[X] {exc}", file=sys.stderr)
      return 1
    case_dir = args.output

  if case_dir is None:
    parser.error("Provide case_dir or use --preset/--case-name with -o")

  case_dir = case_dir.resolve()
  if not case_dir.is_dir():
    print(f"[X] Not a directory: {case_dir}", file=sys.stderr)
    return 1

  if args.extract_only:
    print(f"[✓] Extract-only complete: {case_dir}")
    return 0

  if not args.skip_translate:
    print(f"\n[translate] ESI middleware on {case_dir}")
    try:
      ESITranslator(case_dir).run_translation_pipeline()
    except ValueError as exc:
      print(f"[X] Translation blocked: {exc}", file=sys.stderr)
      return 1

  if args.run:
    try:
      application = _read_application(case_dir)
      _run_openfoam(case_dir, application, timeout=args.timeout)
    except subprocess.CalledProcessError as exc:
      print(f"[X] OpenFOAM command failed (exit {exc.returncode})", file=sys.stderr)
      return 1
    except subprocess.TimeoutExpired:
      print("[X] OpenFOAM command timed out", file=sys.stderr)
      return 1
    except (FileNotFoundError, ValueError) as exc:
      print(f"[X] {exc}", file=sys.stderr)
      return 1
    print(f"\n[✓] OpenFOAM run finished for {application}")

  print(f"\n[✓] Verification complete: {case_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
