"""
Runtime middleware: translate Foundation OpenFOAM v10 case syntax to ESI (v2312/v2512).
"""

from __future__ import annotations
from logger import print

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from openfoam_target import uses_legacy_esi_translation
from utils import read_case_text, save_case_text, case_file_key, validate_case_text_target, is_solver_output_dir, is_initial_time_dir, run_openfoam_utility
from services.case_paths import safe_case_path
from services.solver_execution import (
    _application_name, application_entry, application_requires_native, read_case_application,
)
from services.case_inputs import dictionary_code, dictionary_entries, dictionary_primitive_entry, native_foamfile_header

_DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parent / "esi_translation_rules.json"
)

_SKIP_FILENAMES = {".", "..", "Allrun", "Allmesh", "Allclean"}
_SKIP_SUFFIXES = ("~", ".bak", ".orig", ".sh", ".py", ".out", ".err", ".log")
_BINARY_EXTENSIONS = {
    ".stl", ".zip", ".foam", ".vtk", ".obj", ".ans", ".dat",
    ".emesh", ".face", ".edge", ".points", ".faces", ".owner", ".neighbour",
}
_HEADER_CLASSES = {
    name: "dictionary" for name in (
        "blockMeshDict", "controlDict", "fvSchemes", "fvSolution", "thermophysicalProperties",
        "turbulenceProperties", "physicalProperties", "momentumTransport", "transportProperties")
} | {"U": "volVectorField", "p": "volScalarField"}


def _strip_markdown_fences(content):
    content = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", content)
    return re.sub(r"\s*```$", "", content).strip()


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
        self._files = {}
        shadowed = set()
        for path in self._iter_case_files():
            if path.suffix == '.gz' and path.with_suffix('').is_file():
                shadowed.add(path)
                continue
            try:
                content = read_case_text(path)
            except UnicodeDecodeError:
                continue
            relative = path.relative_to(self.case_path)
            known = (len(relative.parts) > 1 and case_file_key(path.name) in _HEADER_CLASSES
                     and (relative.parts[0] in {'system', 'constant', '0.orig'}
                          or is_initial_time_dir(relative.parts[0])))
            if known or re.match(r'\s*FoamFile\s*\{', dictionary_code(content)):
                self._files[path] = content
        original = dict(self._files)
        self._defensive_solver_intercept()
        self._check_indirect_property_entries()
        legacy_turbulence_source, source_paths = self._remap_structure()
        self._translate_all_files(legacy_turbulence_source)
        self._provision_transport_properties(legacy_turbulence_source, original, source_paths)
        self._check_unsupported_function_objects()
        self._inject_fv_solution_macros()
        removed = original.keys() - self._files.keys()
        removed |= {path for path in shadowed if path.with_suffix('') in removed}
        # Validate the complete conversion before touching any original input.
        for path in self._files.keys() | removed:
            relative = path.relative_to(self.case_path)
            safe_case_path(self.case_path, str(relative.parent), relative.name)
            validate_case_text_target(path)
        for path, content in self._files.items():
            if original.get(path) != content:
                save_case_text(path, content)
        for path in removed:
            path.unlink()
        print("<esi_patch>ESI translation complete</esi_patch>")

    def _path(self, relative):
        plain = self.case_path / relative
        compressed = Path(str(plain) + '.gz')
        return plain if plain in self._files or compressed not in self._files else compressed

    # ------------------------------------------------------------------
    # Algorithm 1: defensive solver intercept
    # ------------------------------------------------------------------
    def _defensive_solver_intercept(self) -> None:
        solver = self._application = self._read_application()
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
    def _remap_structure(self) -> tuple[Path | None, dict[Path, Path]]:
        source_paths = {path: path for path in self._files}
        mapped_names = {Path(source).name for source in self.rules.get("file_maps", {})
                        if Path(source).parent == Path("constant")}
        regional_sources = []
        for path in self._files:
            relative = Path(case_file_key(path.relative_to(self.case_path)))
            if (len(relative.parts) == 3 and relative.parts[0] == "constant"
                    and relative.name in mapped_names):
                regional_sources.append((path, relative.parts[1]))
        properties = self.case_path / "constant/regionProperties"
        if regional_sources and (properties.is_file() or properties.with_suffix(".gz").is_file()):
            from services.meshing import _active_mesh_directories
            try:
                regions = _active_mesh_directories(self.case_path)
            except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
                details = '\n'.join(str(part) for part in
                                    (error, getattr(error, 'stdout', ''), getattr(error, 'stderr', '')) if part)
                raise ValueError(
                    "Cannot establish active regions for legacy ESI conversion: "
                    f"{details}; inputs preserved") from error
            unsupported = [str(path.relative_to(self.case_path)) for path, region in regional_sources
                           if region in regions]
            if unsupported:
                raise ValueError(
                    "Unsupported regional legacy ESI conversion of "
                    f"{', '.join(unsupported)}; inputs preserved")

        legacy_turbulence_source = None
        for old_rel, new_rel in self.rules.get("file_maps", {}).items():
            old_path = self._path(old_rel)
            new_path = self._path(new_rel)
            if old_path not in self._files:
                continue
            if new_path in self._files:
                raise ValueError(f"Cannot establish legacy ESI conversion with both {old_rel} and {new_rel}; inputs preserved")
            if old_path.suffix == '.gz':
                new_path = Path(str(new_path) + '.gz')
            self._files[new_path] = self._files.pop(old_path)
            source_paths[new_path] = source_paths.pop(old_path)
            if (case_file_key(old_path) == str(self.case_path / 'constant/momentumTransport')
                    and case_file_key(new_path) == str(self.case_path / 'constant/turbulenceProperties')):
                legacy_turbulence_source = source_paths[new_path]
        return legacy_turbulence_source, source_paths

    # ------------------------------------------------------------------
    # Algorithm 3: per-file sanitize + regex keyword swaps
    # ------------------------------------------------------------------
    def _translate_all_files(self, legacy_turbulence_source: Path | None) -> None:
        for file_path in list(self._files):
            self._translate_file(file_path, legacy_turbulence_source)

    def _iter_case_files(self) -> list[Path]:
        paths: list[Path] = []
        for root, dirs, files in os.walk(self.case_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != 'polyMesh'
                       and not (Path(root) == self.case_path and is_solver_output_dir(d))]
            for name in files:
                if name in _SKIP_FILENAMES or name.endswith(_SKIP_SUFFIXES):
                    continue
                path = Path(root) / name
                if path.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                if path.is_symlink() and case_file_key(name) not in _HEADER_CLASSES:
                    continue
                paths.append(path)
        return paths

    def _translate_file(self, file_path: Path, legacy_turbulence_source: Path | None) -> None:
        original = self._files[file_path]

        modified = self._sanitize_llm_artifacts(original, file_path)
        if file_path == self._path('system/controlDict'):
            modified = self._translate_application(modified)
        modified = self._translate_property_entries(modified, file_path)
        if case_file_key(file_path.name) == 'turbulenceProperties':
            modified = self._translate_turbulence_model(
                modified, file_path,
                legacy_turbulence_source if file_path == self._path('constant/turbulenceProperties') else None)

        if modified != original:
            self._files[file_path] = modified

    def _translate_turbulence_model(
        self, content: str, file_path: Path, legacy_source: Path | None,
    ) -> str:
        simulation = dictionary_primitive_entry(content, 'simulationType')
        if simulation is None:
            if legacy_source is not None:
                raise ValueError(
                    f"Cannot establish legacy ESI conversion in {legacy_source.relative_to(self.case_path)}: "
                    "simulationType is not a direct primitive entry; inputs preserved")
            return content
        literal = re.fullmatch(r'"(laminar|RAS|LES)"|(laminar|RAS|LES)', simulation[1].strip())
        if literal is None:
            if legacy_source is not None:
                raise ValueError(
                    f"Cannot establish legacy ESI conversion in {legacy_source.relative_to(self.case_path)}: "
                    f"simulationType {simulation[1].strip()!r} does not establish a supported literal family; "
                    "indirect family selection is not resolved; inputs preserved")
            return content
        family = literal[1] or literal[2]
        indirect_root = any(key[0].startswith(('#', '$', '"'))
                            for key, _, _ in dictionary_entries(content))

        def require_native_value(entry, expected):
            source = legacy_source or file_path
            selected = self._query_native_dictionary(source, entry)
            key = re.escape(entry.rsplit('/', 1)[-1])
            actual = re.search(rf'(?:^|\n)(?:{key}|"(?:\\.|[^"\\\n])+")\s+([^;\n]*);\s*\Z',
                               selected.stdout)
            if actual is None or actual[1].strip().strip('"') != expected:
                raise ValueError(
                    f"Cannot establish legacy ESI conversion in {source.relative_to(self.case_path)}: "
                    f"preprocessed {entry} does not preserve the direct selector {expected!r}; "
                    f"native output: {selected.stdout}; inputs preserved")

        if family == 'laminar':
            if legacy_source is not None and indirect_root:
                require_native_value('simulationType', family)
            return content
        span = _find_dict_block(content, family)
        if span is None:
            if legacy_source is not None:
                raise ValueError(
                    f"Cannot establish legacy ESI conversion in {legacy_source.relative_to(self.case_path)}: "
                    f"active {family} dictionary is not directly available; inputs preserved")
            return content
        start, end = span
        opening = dictionary_code(content, mask_strings=True).index('{', start)
        body = content[opening + 1:end - 1]
        model = dictionary_primitive_entry(body, 'model')
        if indirect_root and (legacy_source is not None or model is not None):
            require_native_value('simulationType', family)
        if model is None:
            if legacy_source is not None and dictionary_primitive_entry(body, family + 'Model') is None:
                raise ValueError(
                    f"Cannot establish legacy ESI conversion in {legacy_source.relative_to(self.case_path)}: "
                    f"active {family} model selector is not a direct model or {family}Model entry; "
                    "included selectors are not converted; inputs preserved")
            return content
        if dictionary_primitive_entry(body, family + 'Model') is not None:
            raise ValueError(
                f"Cannot establish legacy ESI conversion in {file_path.relative_to(self.case_path)}: "
                f"{family} contains both model and {family}Model; inputs preserved")
        if indirect_root or any(key[0].startswith(('#', '$', '"'))
                                for key, _, _ in dictionary_entries(body)):
            selected = re.fullmatch(r'"([^"\\\n]+)"|([^\s;{}()$#"]+)', model[1].strip())
            source = legacy_source or file_path
            if selected is None:
                raise ValueError(
                    f"Cannot establish legacy ESI conversion in {source.relative_to(self.case_path)}: "
                    f"indirect {family}/model cannot be safely renamed; inputs preserved")
            require_native_value(f'{family}/model', selected[1] or selected[2])
            if self._query_native_dictionary(source, f'{family}/{family}Model', allow_absent=True) is not None:
                raise ValueError(
                    f"Cannot establish legacy ESI conversion in {source.relative_to(self.case_path)}: "
                    f"preprocessed {family}/{family}Model conflicts with renaming model; inputs preserved")
        key_end = opening + 1 + model.start()
        return content[:key_end - len('model')] + family + 'Model' + content[key_end:]

    def _sanitize_llm_artifacts(self, content: str, file_path: Path) -> str:
        content = _strip_markdown_fences(content)

        if re.match(r'\s*FoamFile\s*\{', dictionary_code(content)):
            header = native_foamfile_header(content)
            if header is not None and dictionary_primitive_entry(header[0], 'version') is None:
                # Foundation v10 defaults omitted versions to 2.0; ESI requires the entry.
                opening = header[1] - 1 - len(header[0])
                content = content[:opening] + '\n    version     2.0;' + content[opening:]
            return content

        file_name = case_file_key(file_path.name)
        folder_name = file_path.parent.name
        obj_class = _HEADER_CLASSES[file_name]
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

    def _query_native_dictionary(self, file_path: Path, entry: str, *, keywords=False, allow_absent=False):
        expand = os.getenv('WM_PROJECT_VERSION') == '10'
        relative = case_file_key(file_path.relative_to(self.case_path))
        command = ['foamDictionary', *(['-expand'] if expand else []),
                   relative, '-entry', entry, *(['-keywords'] if keywords else [])]
        try:
            return run_openfoam_utility(command, working_dir=str(self.case_path), timeout=3600)
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            diagnostic = str(getattr(error, 'stderr', '') or '')
            # This is foamDictionary's missing-entry branch, not a dictionary parse error.
            if (allow_absent and isinstance(error, subprocess.CalledProcessError)
                    and 'FOAM FATAL IO ERROR:' in diagnostic
                    and re.search(rf'^Cannot find entry {re.escape(entry)}(?: in file {re.escape(relative)})?\s*$',
                                  diagnostic, re.M)
                    and re.search(r'From (?:function )?int main\([^\n]*\)\s*\n\s*in file foamDictionary\.C\b',
                                  diagnostic)):
                return None
            details = '\n'.join(str(part) for part in
                                (error, getattr(error, 'stdout', ''), getattr(error, 'stderr', '')) if part)
            raise ValueError(
                f"Cannot establish native legacy ESI entries in {file_path.relative_to(self.case_path)}: "
                f"{details}; inputs preserved") from error

    def _native_dictionary_keys(self, file_path: Path, entry: str) -> set[str]:
        result = self._query_native_dictionary(file_path, entry, keywords=True)
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def _check_indirect_property_entries(self) -> None:
        for file_path, content in self._files.items():
            if file_path.relative_to(self.case_path).parts[0] != 'constant':
                continue
            scoped = list(dictionary_entries(content))
            for keyword, rule in self.rules.get('property_entry_maps', {}).items():
                if case_file_key(file_path.name) not in rule['files']:
                    continue
                if any(key[0] == f'"{keyword}"' for key, _, _ in scoped):
                    continue  # The existing exact-quoted conversion guard rejects this before writes.
                value = dictionary_primitive_entry(content, keyword)
                if value is not None and ('$' in value[1] or value[1].lstrip().startswith('#')):
                    raise ValueError(
                        f"Cannot establish legacy ESI property conversion in {file_path.relative_to(self.case_path)}: "
                        f"indirect {keyword} selector is not supported; inputs preserved")
                replacement = rule['replacement']
                if value is not None and any(key[0] in {replacement, f'"{replacement}"'}
                                             for key, _, _ in scoped):
                    raise ValueError(
                        f"Cannot establish legacy ESI property conversion in {file_path.relative_to(self.case_path)}: "
                        f"both {keyword} and {replacement} selectors are supplied; inputs preserved")
                if not any(key[0].startswith(('#', '$', '"')) for key, _, _ in scoped):
                    continue
                selected = self._query_native_dictionary(file_path, keyword, allow_absent=True)
                if selected is not None:
                    # Native expansion diagnostics precede the terminal named entry.
                    actual = re.search(rf'(?:^|\n){re.escape(keyword)}\s+([^;\n]*);\s*\Z', selected.stdout)
                    if value is None or actual is None or actual[1].strip() != value[1].strip():
                        raise ValueError(
                            f"Cannot establish legacy ESI property conversion in {file_path.relative_to(self.case_path)}: "
                            f"preprocessed {keyword} selector cannot be safely rewritten; inputs preserved")
                if value is not None and self._query_native_dictionary(file_path, replacement, allow_absent=True) is not None:
                    raise ValueError(
                        f"Cannot establish legacy ESI property conversion in {file_path.relative_to(self.case_path)}: "
                        f"preprocessed {replacement} selector conflicts with {keyword}; inputs preserved")

    def _translate_property_entries(self, content: str, file_path: Path) -> str:
        if file_path.relative_to(self.case_path).parts[0] != 'constant':
            return content
        for keyword, rule in self.rules.get('property_entry_maps', {}).items():
            if case_file_key(file_path.name) not in rule['files']:
                continue
            code = dictionary_code(content)
            scoped = list(dictionary_entries(content))
            if any(entry[0] == f'"{keyword}"' for entry, _, _ in scoped):
                raise ValueError(
                    f"Cannot establish legacy ESI property conversion in {file_path.relative_to(self.case_path)}: "
                    f"quoted {keyword} selector is not a supported direct entry; inputs preserved")
            entries = [entry for entry, start, end in scoped
                       if entry[0] == keyword and end is not None
                       and code[start:start + 1] != '{'
                       and code[entry.end():end - 1].strip()]
            for entry in reversed(entries):
                content = content[:entry.start()] + rule['replacement'] + content[entry.end():]
        return content

    # ------------------------------------------------------------------
    # Algorithm 4: fvSolution pFinal compatibility
    # ------------------------------------------------------------------
    def _inject_fv_solution_macros(self) -> None:
        fv_solution = self._path("system/fvSolution")
        if fv_solution not in self._files:
            return

        solver = self._application
        if solver not in set(self.rules.get("transient_solvers", [])):
            return

        content = self._files[fv_solution]
        span = _find_dict_block(content, 'solvers')
        if span is None:
            return
        start, end = span
        opening = dictionary_code(content, mask_strings=True).index('{', start)
        solvers = content[opening + 1:end - 1]
        if any(_find_dict_block(solvers, key) is not None
               for key in ('pFinal', '"pFinal"')):
            return
        if any(key[0].startswith(('#', '$', '"')) for key, _, _ in dictionary_entries(solvers)):
            keys = self._native_dictionary_keys(fv_solution, 'solvers')
            if keys & {'pFinal', '"pFinal"'}:
                return
            if any(key.startswith('"') for key in keys):
                # Let native matching decide whether a supplied pattern selects pFinal.
                if self._query_native_dictionary(fv_solution, 'solvers/pFinal', allow_absent=True) is not None:
                    return
        updated = self._inject_pfinal_block(solvers)
        if updated != solvers:
            self._files[fv_solution] = content[:opening + 1] + updated + content[end - 1:]

    def _read_application(self) -> str | None:
        control_dict = self._path("system/controlDict")
        if control_dict not in self._files:
            return None
        try:
            content = _strip_markdown_fences(self._files[control_dict])
            value = application_entry(content)
            if value is None:
                sources = [source for source in self.rules.get('file_maps', {})
                           if self._path(source) in self._files]
                if sources:
                    raise ValueError(
                        "application is not a direct primitive entry while legacy remapping of "
                        f"{', '.join(sources)} requires solver-dependent reconciliation")
                return None
            literal = _application_name(value[1])
            if application_requires_native(content):
                actual = _application_name(read_case_application(str(self.case_path)))
                if actual != literal:
                    raise ValueError(
                        f"native application {actual!r} differs from direct literal {literal!r}; "
                        "preprocessed solver selection cannot be converted by editing that literal")
                if literal in self.rules.get('application_maps', {}):
                    raise ValueError(
                        f"application {actual!r} requires a rename, but preprocessing prevents "
                        "establishing that editing its direct literal changes the effective selection")
            return literal
        except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
            details = '\n'.join(str(part) for part in
                                (error, getattr(error, 'stdout', ''), getattr(error, 'stderr', '')) if part)
            raise ValueError(
                f"Cannot establish solver-dependent legacy ESI conversion from "
                f"system/controlDict application: {details}; inputs preserved") from error

    def _translate_application(self, content):
        replacement = self.rules.get('application_maps', {}).get(self._application)
        if replacement is None:
            return content
        value = application_entry(content)
        start = value.start(1) + len(value[1]) - len(value[1].lstrip())
        end = value.start(1) + len(value[1].rstrip())
        if value[1].strip().startswith('"'):
            replacement = '"' + replacement + '"'
        self._application = _application_name(replacement)
        return content[:start] + replacement + content[end:]

    def _inject_pfinal_block(self, content: str) -> str:
        pfinal = self.rules.get("pfinal_block", "")
        if not pfinal:
            return content

        for entry in ("p", '"p.*"'):
            span = _find_dict_block(content, entry)
            if span is None:
                continue
            _, end = span
            return content[:end] + pfinal + content[end:]

        return content

    def _translate_transport_model_value(self, content, source, original):
        model = dictionary_primitive_entry(content, 'transportModel')
        supplied_model = dictionary_primitive_entry(original, 'transportModel')
        literal_constant = model is not None and model[1].strip() in ('constant', '"constant"')
        # A property keyword rename has already established the actual source and
        # absence of a conflicting transportModel before the structural remap.
        if model is not None and supplied_model is None:
            return _replace_primitive_value(content, model, 'Newtonian') if literal_constant else content
        scoped = list(dictionary_entries(original))
        preprocessing = any(key[0].startswith(('#', '$')) for key, _, _ in scoped)
        indirect = (preprocessing or any(key[0].startswith('"') for key, _, _ in scoped)
                    or supplied_model is not None and ('$' in supplied_model[1]
                        or supplied_model[1].lstrip().startswith('#')))
        if not indirect:
            return _replace_primitive_value(content, model, 'Newtonian') if literal_constant else content
        selected = self._query_native_dictionary(source, 'transportModel', allow_absent=True)
        if selected is None:
            return content
        actual = re.search(r'(?:^|\n)(transportModel|"(?:\\.|[^"\\\n])+")\s+([^;\n]*);\s*\Z',
                           selected.stdout)
        if actual is None:
            raise ValueError(
                f"Cannot establish legacy ESI transportModel conversion in {source.relative_to(self.case_path)}: "
                f"native output is not a selected primitive entry: {selected.stdout}; inputs preserved")
        if actual[2].strip() not in ('constant', '"constant"'):
            return content
        if literal_constant and not preprocessing and actual[1] == 'transportModel':
            return _replace_primitive_value(content, model, 'Newtonian')
        raise ValueError(
            f"Cannot establish legacy ESI transportModel conversion in {source.relative_to(self.case_path)}: "
            "effective constant selector requires Newtonian but its indirect or quoted transportModel definition "
            "cannot be safely edited; "
            "inputs preserved")

    def _provision_transport_properties(self, legacy_turbulence_source, original, source_paths) -> None:
        """ESI incompressible solvers (e.g. icoFoam) read constant/transportProperties."""
        solver = self._application
        if solver not in set(self.rules.get("solvers_using_transport_properties", [])):
            return

        transport = self._path("constant/transportProperties")
        if transport in self._files:
            if legacy_turbulence_source is not None:
                source = source_paths[transport]
                self._files[transport] = self._translate_transport_model_value(
                    self._files[transport], source, original[source])
            return

        for source_name in ("thermophysicalProperties", "physicalProperties"):
            source = self._path("constant/" + source_name)
            if source not in self._files:
                continue
            content = self._files[source]
            if dictionary_primitive_entry(content, 'nu') is None:
                raise ValueError(
                    f"Cannot establish legacy ESI constant/transportProperties conversion from {source_name}: "
                    "no supplied nu entry; inputs preserved, no physical defaults were generated")
            header = _find_dict_block(content, 'FoamFile')
            if header is not None:
                start, end = header
                opening = dictionary_code(content, mask_strings=True).index('{', start)
                body = content[opening + 1:end - 1]
                object_entry = dictionary_primitive_entry(body, 'object')
                if object_entry is not None:
                    body = _replace_primitive_value(body, object_entry, 'transportProperties')
                    content = content[:opening + 1] + body + content[end - 1:]
            if self._path("constant/turbulenceProperties") in self._files:
                native_source = source_paths[source]
                content = self._translate_transport_model_value(content, native_source, original[native_source])
            if source.suffix == '.gz':
                transport = Path(str(transport) + '.gz')
            self._files[transport] = content
            return

    def _check_unsupported_function_objects(self) -> None:
        control_dict = self._path("system/controlDict")
        if control_dict not in self._files:
            return
        content = self._files[control_dict]
        for key, start, _ in dictionary_entries(content, recursive=True):
            if key[0] == '#includeFunc' or (key[0] == 'cacheTemporaryObjects'
                                           and content[start:start + 1] == '('):
                raise ValueError(f"Unsupported legacy ESI conversion in system/controlDict: {key[0]}; inputs preserved")


def _replace_primitive_value(content, entry, replacement):
    value = entry[1]
    start = entry.start(1) + len(value) - len(value.lstrip())
    end = entry.start(1) + len(value.rstrip())
    if value.strip().startswith('"'):
        replacement = '"' + replacement + '"'
    return content[:start] + replacement + content[end:]


def _find_dict_block(content: str, entry_name: str) -> tuple[int, int] | None:
    """Find a direct dictionary entry in the supplied native dictionary scope."""
    for key, start, end in dictionary_entries(content):
        if key[0] == entry_name and content[start:start + 1] == '{' and end is not None:
            return key.start(), end
    return None


def convert_case_to_esi_if_needed(case_dir: str | Path, config: Any) -> None:
    """Run the legacy ESI translation path when explicitly using generic ESI.

    Explicit native targets use their own generation/runtime contracts and
    bypass this Foundation-v10-to-ESI middleware, even when the historical
    ``openfoam_fork=esi`` setting is also present.
    """
    if not uses_legacy_esi_translation(config):
        return

    rules_path = getattr(config, "esi_translation_rules_path", None) or _DEFAULT_RULES_PATH
    ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()
