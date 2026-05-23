"""
Runtime middleware: translate Foundation OpenFOAM v10 case syntax to ESI (v2312/v2512).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "esi_translation_rules.json"
)

_SKIP_FILENAMES = {".", ".."}
_SKIP_SUFFIXES = ("~", ".bak", ".orig")
_BINARY_EXTENSIONS = {
    ".stl", ".gz", ".zip", ".foam", ".vtk", ".obj", ".ans", ".dat",
    ".eMesh", ".face", ".edge", ".points", ".faces", ".owner", ".neighbour",
}


class ESITranslator:
    """Translate a generated Foundation v10 case directory for ESI OpenFOAM."""

    def __init__(
        self,
        case_path: str | Path,
        rules: dict[str, Any] | None = None,
        rules_path: str | Path | None = None,
    ) -> None:
        self.case_path = Path(case_path).resolve()
        if rules is not None:
            self.rules = rules
        else:
            path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
            with open(path, encoding="utf-8") as f:
                self.rules = json.load(f)

        self._application: str | None = None

    def run_translation_pipeline(self) -> None:
        print(f"<esi_patch>Booting ESI translation for: {self.case_path}</esi_patch>")
        self._defensive_solver_intercept()
        self._remap_structure()
        self._translate_all_files()
        self._fix_physical_properties()
        self._provision_transport_properties()
        self._sanitize_foundation_function_objects()
        self._inject_fv_solution_macros()
        print("<esi_patch>ESI translation complete</esi_patch>")

    # ------------------------------------------------------------------
    # Algorithm 1: defensive solver intercept
    # ------------------------------------------------------------------
    def _defensive_solver_intercept(self) -> None:
        control_dict = self.case_path / "system" / "controlDict"
        if not control_dict.is_file():
            return

        content = control_dict.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"application\s+(\w+)\s*;", content)
        if not match:
            return

        solver = match.group(1)
        self._application = solver
        blacklist = set(self.rules.get("blacklisted_solvers", []))
        if solver in blacklist:
            raise ValueError(
                f"Solver '{solver}' is not available in ESI OpenFOAM. "
                f"The agent generated a Foundation v10 case that cannot be translated "
                f"for this physics model. Choose a different tutorial/solver or run with "
                f"FOAMAGENT_OPENFOAM_FORK=foundation."
            )

    # ------------------------------------------------------------------
    # Algorithm 2: structural file renames
    # ------------------------------------------------------------------
    def _remap_structure(self) -> None:
        for old_rel, new_rel in self.rules.get("file_maps", {}).items():
            old_path = self.case_path / old_rel
            new_path = self.case_path / new_rel
            if not old_path.is_file():
                continue
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if new_path.exists():
                print(f"<esi_patch>Skip rename (target exists): {old_rel} -> {new_rel}</esi_patch>")
                continue
            old_path.rename(new_path)
            print(f"<esi_patch>Renamed {old_rel} -> {new_rel}</esi_patch>")

    # ------------------------------------------------------------------
    # Algorithm 3: per-file sanitize + regex keyword swaps
    # ------------------------------------------------------------------
    def _translate_all_files(self) -> None:
        for file_path in self._iter_case_files():
            self._translate_file(file_path)

    def _iter_case_files(self) -> list[Path]:
        paths: list[Path] = []
        for root, dirs, files in os.walk(self.case_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if name in _SKIP_FILENAMES or name.endswith(_SKIP_SUFFIXES):
                    continue
                path = Path(root) / name
                if path.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                paths.append(path)
        return paths

    def _translate_file(self, file_path: Path) -> None:
        try:
            original = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return

        modified = self._sanitize_llm_artifacts(original, file_path)
        modified = self._apply_keyword_swaps(modified, file_path)

        if modified != original:
            file_path.write_text(modified, encoding="utf-8")
            rel = file_path.relative_to(self.case_path)
            print(f"<esi_patch>Translated {rel}</esi_patch>")

    def _sanitize_llm_artifacts(self, content: str, file_path: Path) -> str:
        content = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        if "FoamFile" in content:
            return content

        file_name = file_path.name
        folder_name = file_path.parent.name
        class_map = {
            "blockMeshDict": "dictionary",
            "controlDict": "dictionary",
            "fvSchemes": "dictionary",
            "fvSolution": "dictionary",
            "thermophysicalProperties": "dictionary",
            "turbulenceProperties": "dictionary",
            "physicalProperties": "dictionary",
            "momentumTransport": "dictionary",
            "transportProperties": "dictionary",
            "U": "volVectorField",
            "p": "volScalarField",
        }
        obj_class = class_map.get(file_name, "dictionary")
        header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {obj_class};
    location    "{folder_name}";
    object      {file_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
        print(
            f"<esi_patch>Injected FoamFile header into "
            f"{file_path.relative_to(self.case_path)}</esi_patch>"
        )
        return header + content

    def _apply_keyword_swaps(self, content: str, file_path: Path) -> str:
        for rule in self.rules.get("keyword_swaps", []):
            scope = rule.get("scope")
            if scope and file_path.name not in scope:
                continue
            pattern = rule["pattern"]
            replacement = rule["replacement"]
            content = re.sub(pattern, replacement, content)
        return content

    # ------------------------------------------------------------------
    # Algorithm 4: fvSolution injections (pFinal, pRef)
    # ------------------------------------------------------------------
    def _inject_fv_solution_macros(self) -> None:
        fv_solution = self.case_path / "system" / "fvSolution"
        if not fv_solution.is_file():
            return

        solver = self._application or self._read_application()
        if solver not in set(self.rules.get("transient_solvers", [])):
            return

        content = fv_solution.read_text(encoding="utf-8", errors="ignore")
        updated = content

        if "pFinal" not in updated:
            updated = self._inject_pfinal_block(updated)

        if solver in set(self.rules.get("incompressible_transient_solvers", [])):
            updated = self._inject_pref_in_piso(updated)

        if updated != content:
            fv_solution.write_text(updated, encoding="utf-8")
            print("<esi_patch>Updated system/fvSolution for ESI transient solvers</esi_patch>")

    def _read_application(self) -> str | None:
        control_dict = self.case_path / "system" / "controlDict"
        if not control_dict.is_file():
            return None
        match = re.search(
            r"application\s+(\w+)\s*;",
            control_dict.read_text(encoding="utf-8", errors="ignore"),
        )
        return match.group(1) if match else None

    def _inject_pfinal_block(self, content: str) -> str:
        pfinal = self.rules.get("pfinal_block", "")
        if not pfinal:
            return content

        for entry in ("p", '"p.*"'):
            span = _find_dict_block(content, entry)
            if span is None:
                continue
            start, end = span
            block = content[start:end]
            return content[:end] + pfinal + content[end:]

        return content

    def _inject_pref_in_piso(self, content: str) -> str:
        if "pRefCell" in content:
            return content

        span = _find_dict_block(content, "PISO")
        if span is None:
            span = _find_dict_block(content, "PIMPLE")
        if span is None:
            return content

        start, end = span
        block = content[start:end]
        if "pRefCell" in block:
            return content

        injection = self.rules.get("pref_injection", "")
        if not injection:
            return content

        # Insert before closing brace of PISO/PIMPLE block
        closing = block.rfind("}")
        if closing == -1:
            return content
        new_block = block[:closing] + injection + "\n" + block[closing:]
        return content[:start] + new_block + content[end:]

    # ------------------------------------------------------------------
    # Algorithm 5: physical properties cleanup (RAG Chemkin artifacts)
    # ------------------------------------------------------------------
    def _fix_physical_properties(self) -> None:
        candidates = [
            self.case_path / "constant" / "thermophysicalProperties",
            self.case_path / "constant" / "physicalProperties",
            self.case_path / "constant" / "transportProperties",
        ]
        markers = self.rules.get("chemkin_markers", [])

        for path in candidates:
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if not any(m in content for m in markers):
                # Ensure laminar incompressible nu block when only bad transport leaked
                if path.name == "transportProperties" and "nu" not in content:
                    continue
                if path.name in ("physicalProperties", "thermophysicalProperties"):
                    if "nu" in content and not any(m in content for m in markers):
                        continue
                continue

            nu_block = (
                "FoamFile\n"
                "{\n"
                "    version     2.0;\n"
                "    format      ascii;\n"
                "    class       dictionary;\n"
                f"    object      {path.name};\n"
                "}\n\n"
                "nu              0.01;\n"
            )
            path.write_text(nu_block, encoding="utf-8")
            rel = path.relative_to(self.case_path)
            print(f"<esi_patch>Reset Chemkin-corrupted properties in {rel}</esi_patch>")

    def _provision_transport_properties(self) -> None:
        """ESI incompressible solvers (e.g. icoFoam) read constant/transportProperties."""
        solver = self._application or self._read_application()
        if solver not in set(self.rules.get("solvers_using_transport_properties", [])):
            return

        transport = self.case_path / "constant" / "transportProperties"
        if transport.is_file():
            return

        for source_name in ("thermophysicalProperties", "physicalProperties"):
            source = self.case_path / "constant" / source_name
            if not source.is_file():
                continue
            content = source.read_text(encoding="utf-8", errors="ignore")
            content = re.sub(
                r"(object\s+)\S+\s*;",
                r"\1transportProperties;",
                content,
                count=1,
            )
            if (self.case_path / "constant" / "turbulenceProperties").is_file():
                content = re.sub(
                    r"transportModel\s+constant\s*;",
                    "transportModel  Newtonian;",
                    content,
                )
            transport.write_text(content, encoding="utf-8")
            print(
                f"<esi_patch>Created constant/transportProperties from {source_name}</esi_patch>"
            )
            return

    def _sanitize_foundation_function_objects(self) -> None:
        """Remove Foundation-only #includeFunc blocks that ESI controlDict cannot parse."""
        control_dict = self.case_path / "system" / "controlDict"
        if not control_dict.is_file():
            return

        content = control_dict.read_text(encoding="utf-8", errors="ignore")
        if "#includeFunc" not in content and "cacheTemporaryObjects" not in content:
            return

        updated = _remove_dict_block(content, "functions")
        updated = re.sub(
            r"cacheTemporaryObjects\s*\([^;]*\)\s*;?",
            "// ESI: removed cacheTemporaryObjects\n",
            updated,
            flags=re.DOTALL,
        )
        if updated != content:
            control_dict.write_text(updated, encoding="utf-8")
            print("<esi_patch>Removed Foundation functionObjects from controlDict</esi_patch>")


def _remove_dict_block(content: str, entry_name: str) -> str:
    """Remove a top-level dictionary entry and its { ... } body."""
    span = _find_dict_block(content, entry_name)
    if span is None:
        return content
    start, end = span
    return content[:start] + f"// ESI: removed Foundation v10 '{entry_name}' block\n" + content[end:]


def _find_dict_block(content: str, entry_name: str) -> tuple[int, int] | None:
    """Return (start, end) slice for `entry_name { ... }` with balanced braces."""
    pattern = rf"(?:^|\n)\s*{re.escape(entry_name)}\s*\{{"
    match = re.search(pattern, content)
    if not match:
        return None

    brace_start = content.find("{", match.start())
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(content)):
        ch = content[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return match.start(), i + 1
    return None


def convert_case_to_esi_if_needed(case_dir: str | Path, config: Any) -> None:
    """Run ESI translation when config.openfoam_fork == 'esi'."""
    fork = getattr(config, "openfoam_fork", "foundation")
    if fork != "esi":
        return

    rules_path = getattr(config, "esi_translation_rules_path", None) or _DEFAULT_RULES_PATH
    ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()
