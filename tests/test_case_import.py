"""Tests for the Foundation-v10-only existing-case import workflow."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from services import case_import  # noqa: E402
from services.case_import_allrun import parse_allrun  # noqa: E402
from services.case_import import (  # noqa: E402
    CaseImportError,
    apply_safe_repairs,
    import_case,
)


def _foundation_header() -> str:
    return """/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM: The Open Source CFD Toolbox
| Website:  https://openfoam.org
| Version:  10
\\*---------------------------------------------------------------------------*/
"""


def _write_case(root: Path, *, allrun: str | None = None, esi: bool = False) -> Path:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    header = _foundation_header()
    if esi:
        header = header.replace("https://openfoam.org", "www.openfoam.com").replace(
            "Version:  10", "Version:  v2312"
        )
    (root / "system" / "controlDict").write_text(
        header
        + """FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application icoFoam;
startTime 0;
endTime 2;
""",
        encoding="utf-8",
    )
    (root / "system" / "blockMeshDict").write_text(
        header + "FoamFile { object blockMeshDict; }\n",
        encoding="utf-8",
    )
    if allrun is not None:
        (root / "Allrun").write_text(allrun, encoding="utf-8")
    return root


def test_import_preserves_original_and_inserts_mesh_gate(tmp_path: Path) -> None:
    source = _write_case(
        tmp_path / "source",
        allrun="""#!/bin/sh
cd ${0%/*} || exit 1
. $WM_PROJECT_DIR/bin/tools/RunFunctions
application=$(getApplication)
runApplication blockMesh
runApplication $application
""",
    )

    manifest = import_case(source, tmp_path / "import-output")

    assert manifest.platform == "foundation-v10"
    assert [step.command for step in manifest.execution_plan] == [
        "blockMesh",
        "checkMesh",
        "icoFoam",
    ]
    assert (tmp_path / "import-output" / "original" / "system" / "controlDict").read_text(
        encoding="utf-8"
    ) == (source / "system" / "controlDict").read_text(encoding="utf-8")
    original_control = tmp_path / "import-output" / "original" / "system" / "controlDict"
    assert not (original_control.stat().st_mode & stat.S_IWUSR)
    assert manifest.original_hashes["system/controlDict"]


def test_import_rejects_esi_case_without_running_it(tmp_path: Path) -> None:
    source = _write_case(tmp_path / "esi-source", esi=True)

    manifest = import_case(source, tmp_path / "import-output")

    assert not manifest.supported
    assert manifest.platform == "esi"
    assert any("Only Foundation OpenFOAM v10" in issue for issue in manifest.blocking_issues)


def test_import_overwrite_clears_output_contents_without_removing_root(
    tmp_path: Path,
) -> None:
    source = _write_case(tmp_path / "source")
    output = tmp_path / "import-output"
    # Only a directory previously created by Foam-Agent may be cleared.  This
    # makes --overwrite_output safe if a caller accidentally names a valuable
    # existing directory.
    import_case(source, output)
    stale_file = output / "previous-result"
    stale_file.write_text("stale", encoding="utf-8")

    manifest = import_case(source, output, overwrite=True)

    assert output.is_dir()
    assert not stale_file.exists()
    assert (output / "original" / "system" / "controlDict").is_file()
    assert manifest.output_root == str(output.resolve())


def test_import_overwrite_rejects_unowned_output_directory(tmp_path: Path) -> None:
    source = _write_case(tmp_path / "source")
    output = tmp_path / "unowned-output"
    output.mkdir()
    sentinel = output / "do-not-delete"
    sentinel.write_text("important", encoding="utf-8")

    with pytest.raises(CaseImportError, match="unowned output directory"):
        import_case(source, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important"


def test_import_requires_case_subdir_for_multi_case_zip(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive-root"
    _write_case(archive_root / "first")
    _write_case(archive_root / "second")
    archive = tmp_path / "cases.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for path in archive_root.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(archive_root))

    with pytest.raises(CaseImportError, match="Multiple OpenFOAM cases"):
        import_case(archive, tmp_path / "multi-output")

    manifest = import_case(
        archive,
        tmp_path / "selected-output",
        case_subdir="second",
    )
    assert manifest.case_root == "second"
    assert manifest.application == "icoFoam"


def test_safe_repair_changes_header_not_numbers(tmp_path: Path) -> None:
    case = _write_case(tmp_path / "case")
    control_dict = case / "system" / "controlDict"
    control_dict.write_text(
        control_dict.read_text(encoding="utf-8").replace(
            "object controlDict;", "object incorrectName;"
        ),
        encoding="utf-8",
    )

    repairs = apply_safe_repairs(case, [{"error_content": "FoamFile object mismatch"}])

    repaired = control_dict.read_text(encoding="utf-8")
    assert "object controlDict;" in repaired
    assert "endTime 2;" in repaired
    assert any(repair["status"] == "applied" for repair in repairs)


def test_import_run_retries_only_after_a_safe_non_numeric_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_case(tmp_path / "source")
    control_dict = source / "system" / "controlDict"
    control_dict.write_text(
        control_dict.read_text(encoding="utf-8").replace(
            "object controlDict;", "object incorrectName;"
        ),
        encoding="utf-8",
    )
    calls = 0

    def simulated_execution(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [{"error_content": "FoamFile object mismatch"}] if calls == 1 else []

    monkeypatch.setattr(case_import, "execute_imported_case", simulated_execution)
    result = case_import.run_imported_case(
        source,
        tmp_path / "import-output",
        max_repairs=1,
    )

    assert result.status == "success"
    assert calls == 2
    assert "endTime 2;" in (tmp_path / "import-output" / "work" / "system" / "controlDict").read_text(
        encoding="utf-8"
    )
    assert any(
        repair["status"] == "applied"
        for repair in result.attempts[0]["repairs"]
    )


def test_allrun_parser_rejects_arbitrary_shell_setup() -> None:
    with pytest.raises(CaseImportError, match="Unsupported shell setup"):
        parse_allrun(
            ". ./untrusted-helper.sh\nrunApplication icoFoam\n",
            "icoFoam",
        )


def test_allrun_parser_prevents_writes_outside_work_copy() -> None:
    # The importer now rejects every -case override, not merely absolute
    # paths: a nested case would otherwise bypass the selected case's
    # platform and preflight validation.
    with pytest.raises(CaseImportError, match="case arguments|Absolute paths"):
        parse_allrun(
            "runApplication blockMesh -case=/tmp/another-case\n"
            "runApplication icoFoam\n",
            "icoFoam",
        )
