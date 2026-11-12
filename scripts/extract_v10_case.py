#!/usr/bin/env python3
"""
Extract a pristine Foundation v10 tutorial case from the Foam-Agent RAG database.

This simulates high-quality LLM output for testing the ESI translation middleware
without invoking an LLM.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from services.output_safety import validate_output_path

_ENV_DB = os.environ.get("FOAMAGENT_DB_PATH")
DEFAULT_DB = Path(_ENV_DB) if _ENV_DB else (ROOT / "database" / "foundation-v10" / "raw" / "openfoam_tutorials_details.txt")

_DIR_LINE = re.compile(
    r"<dir>directory name:\s*(.+?)\.\s*File names in this directory:\s*\[(.*?)\]</dir>"
)
_INDEX_LINE = re.compile(r"^case\s+(\w+):\s*(.+)$")


def extract_case(
    output_dir: str | Path,
    case_name: str,
    *,
    db_path: str | Path = DEFAULT_DB,
    solver: str | None = None,
    domain: str | None = None,
    category: str | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
  """
  Stream the tutorial database and write the first matching case to output_dir.

  Returns the parsed <index> metadata for the matched case.
  """
  db_path = Path(db_path).expanduser().resolve()

  if not db_path.is_file():
    raise FileNotFoundError(f"Database not found: {db_path}")

  output_dir = validate_output_path(output_dir, source_path=db_path)
  if output_dir.exists():
    if not output_dir.is_dir():
      raise NotADirectoryError(f"Output is not a directory: {output_dir}")
    if not overwrite:
      raise FileExistsError(
        f"Output directory already exists: {output_dir} (use --overwrite)"
      )

  output_dir.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=".foamagent-extract-", dir=output_dir.parent) as temporary:
    staging = Path(temporary) / "case"
    metadata = _stream_extract(
      db_path=db_path,
      output_dir=staging,
      case_name=case_name,
      solver=solver,
      domain=domain,
      category=category,
    )
    if metadata is None:
      filters = [f"name={case_name!r}"]
      if solver:
        filters.append(f"solver={solver!r}")
      if domain:
        filters.append(f"domain={domain!r}")
      if category:
        filters.append(f"category={category!r}")
      raise LookupError("No matching complete case found in database (" + ", ".join(filters) + ")")
    backup = Path(temporary) / "previous"
    if output_dir.exists():
      output_dir.rename(backup)
    try:
      staging.rename(output_dir)
    except OSError:
      if backup.exists():
        backup.rename(output_dir)
      raise

  return metadata


def _matches_index(
    index: dict[str, str],
    *,
    case_name: str,
    solver: str | None,
    domain: str | None,
    category: str | None,
) -> bool:
  if index.get("name") != case_name:
    return False
  if solver is not None and index.get("solver") != solver:
    return False
  if domain is not None and index.get("domain") != domain:
    return False
  if category is not None and index.get("category") != category:
    return False
  return True


def _parse_directory_structure(block: str) -> dict[str, list[str]]:
  """Parse <directory_structure> block into folder -> [filenames]."""
  folders: dict[str, list[str]] = {}
  for match in _DIR_LINE.finditer(block):
    folder = match.group(1).strip()
    names = [n.strip() for n in match.group(2).split(",") if n.strip()]
    folders[folder] = names
  return folders


def _stream_extract(
    db_path: Path,
    output_dir: Path,
    case_name: str,
    solver: str | None,
    domain: str | None,
    category: str | None,
) -> dict[str, str] | None:
  in_case = False
  in_index = False
  in_directory_structure = False
  in_tutorials = False
  index: dict[str, str] = {}
  index_lines: list[str] = []
  structure_lines: list[str] = []
  target = False
  current_folder: str | None = None
  current_file: str | None = None
  current_lines: list[str] = []
  files_written = 0

  def _flush_file() -> None:
    nonlocal current_file, current_lines, files_written
    if not target or not current_file or not current_folder:
      current_file = None
      current_lines = []
      return
    folder = PurePosixPath(current_folder.replace("\\", "/"))
    filename = PurePosixPath(current_file.replace("\\", "/"))
    if (folder.is_absolute() or ".." in folder.parts
        or folder.parts and folder.parts[0].endswith(":")
        or filename.is_absolute() or len(filename.parts) != 1
        or filename.name == ".." or filename.name.endswith(":")):
      raise ValueError(f"Unsafe tutorial file path: {current_folder!r}/{current_file!r}")
    dest = output_dir.joinpath(*folder.parts)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename.name
    body = "".join(current_lines).strip()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8') as stream:
      stream.write(body + ("\n" if body else ""))
    print(f"  [+] {current_folder}/{current_file}")
    files_written += 1
    current_file = None
    current_lines = []

  with open(db_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
      stripped = line.strip()

      if stripped == "<case_begin>":
        if target:
          raise ValueError("Matched case is missing its closing case marker")
        in_case = True
        in_index = False
        in_directory_structure = False
        in_tutorials = False
        index = {}
        index_lines = []
        structure_lines = []
        target = False
        current_folder = None
        current_file = None
        current_lines = []
        continue

      if not in_case:
        continue

      if stripped == "</case_end>":
        if target:
          if current_file is not None:
            raise ValueError("Matched case contains an unterminated file content block")
          if files_written == 0:
            raise RuntimeError("Matched case block contained no files")
          return index
        in_case = False
        continue

      if not target:
        if stripped == "<index>":
          in_index = True
          index_lines = []
          continue
        if in_index and stripped == "</index>":
          in_index = False
          for raw in index_lines:
            m = _INDEX_LINE.match(raw.strip())
            if m:
              index[m.group(1)] = m.group(2).strip()
          continue
        if in_index:
          index_lines.append(line)
          continue

        if stripped == "<directory_structure>":
          in_directory_structure = True
          structure_lines = []
          continue
        if in_directory_structure and stripped == "</directory_structure>":
          in_directory_structure = False
          if _matches_index(
            index,
            case_name=case_name,
            solver=solver,
            domain=domain,
            category=category,
          ):
            target = True
            folders = _parse_directory_structure("".join(structure_lines))
            print(
              f"[+] Matched case: name={index.get('name')} "
              f"solver={index.get('solver')} "
              f"domain={index.get('domain')} "
              f"category={index.get('category')}"
            )
            print(f"    directories: {', '.join(sorted(folders))}")
          continue
        if in_directory_structure:
          structure_lines.append(line)
          continue

        if stripped == "<tutorials>":
          in_tutorials = True
          continue

        if not in_tutorials:
          continue

      # Target case: extract files under <tutorials>
      if stripped.startswith("<directory_begin>directory name:"):
        if current_file is not None:
          raise ValueError("Matched case contains an unterminated file content block")
        current_folder = stripped.split("directory name:", 1)[1].strip()
        continue

      if stripped.startswith("<file_begin>file name:"):
        if current_file is not None:
          raise ValueError("Matched case contains an unterminated file content block")
        current_file = stripped.split("file name:", 1)[1].strip()
        current_lines = []
        continue

      if current_file is not None:
        if stripped.startswith("<file_content>"):
          line = line.partition("<file_content>")[2]
        content, closing, _ = line.partition("</file_content>")
        current_lines.append(content)
        if closing:
          _flush_file()

  return None


_PRESETS: dict[str, dict[str, str]] = {
  "cavity": {
    "case_name": "cavity",
    "solver": "icoFoam",
    "category": "cavity",
  },
  "pitzDaily": {
    "case_name": "pitzDaily",
    "solver": "simpleFoam",
  },
  "turbineSiting": {
    "case_name": "turbineSiting",
    "solver": "simpleFoam",
  },
}


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Extract a Foundation v10 tutorial case from the Foam-Agent RAG database."
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    required=True,
    help="Output case directory",
  )
  parser.add_argument(
    "--db",
    type=Path,
    default=DEFAULT_DB,
    help=f"Path to openfoam_tutorials_details.txt (default: {DEFAULT_DB})",
  )
  parser.add_argument("--preset", choices=sorted(_PRESETS), help="Built-in case preset")
  parser.add_argument("--case-name", help="Tutorial case name (index field 'name')")
  parser.add_argument("--solver", help="Solver filter (index field 'solver')")
  parser.add_argument("--domain", help="Domain filter (index field 'domain')")
  parser.add_argument("--category", help="Category filter (index field 'category')")
  parser.add_argument("--overwrite", action="store_true", help="Remove output dir if it exists")
  args = parser.parse_args()

  if args.preset:
    preset = _PRESETS[args.preset]
    case_name = preset["case_name"]
    solver = preset.get("solver")
    domain = preset.get("domain")
    category = preset.get("category")
  else:
    if not args.case_name:
      parser.error("Provide --case-name or --preset")
    case_name = args.case_name
    solver = args.solver
    domain = args.domain
    category = args.category

  print(f"Streaming database: {args.db}")
  print(f"Writing case to: {args.output.resolve()}")

  try:
    meta = extract_case(
      output_dir=args.output,
      case_name=case_name,
      db_path=args.db,
      solver=solver,
      domain=domain,
      category=category,
      overwrite=args.overwrite,
    )
  except (FileNotFoundError, LookupError, FileExistsError, ValueError) as exc:
    print(f"[X] {exc}", file=sys.stderr)
    return 1

  print(f"\n[✓] Extracted {case_name} -> {args.output}")
  print(f"    solver={meta.get('solver')} domain={meta.get('domain')} category={meta.get('category')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
