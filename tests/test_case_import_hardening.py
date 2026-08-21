"""Security and boundary regression tests for existing-case imports.

These tests deliberately exercise inputs that should fail *before* an import
output is modified.  A case archive is untrusted input, and an invalid source
must never turn ``--overwrite_output`` into a way to erase a user's prior
results or the source being imported.
"""

from __future__ import annotations

import stat
import os
import subprocess
import sys
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from services import case_import  # noqa: E402
from services import case_import_source  # noqa: E402
from services.case_import_allrun import (  # noqa: E402
    ensure_mesh_check,
    parse_allrun,
    render_controlled_allrun,
)
from services.case_import import (  # noqa: E402
    CaseImportError,
    ExecutionStep,
    _run_imported_manifest,
    apply_safe_repairs,
    import_case,
)
from services.case_import_repairs import numeric_snapshot  # noqa: E402
from services.output_safety import prepare_output_directory  # noqa: E402


def _foundation_header() -> str:
    return """/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM: The Open Source CFD Toolbox
| Website:  https://openfoam.org
| Version:  10
\\*---------------------------------------------------------------------------*/
"""


def _write_foundation_case(
    root: Path,
    *,
    allrun: str | None = None,
    with_block_mesh: bool = True,
    with_poly_mesh: bool = False,
) -> Path:
    """Create the smallest valid Foundation-v10 import fixture."""
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "system" / "controlDict").write_text(
        _foundation_header()
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
    if with_block_mesh:
        (root / "system" / "blockMeshDict").write_text(
            _foundation_header() + "FoamFile { object blockMeshDict; }\n",
            encoding="utf-8",
        )
    if with_poly_mesh:
        (root / "constant" / "polyMesh").mkdir()
        (root / "constant" / "polyMesh" / "boundary").write_text(
            "0\n(\n)\n", encoding="utf-8"
        )
    if allrun is not None:
        (root / "Allrun").write_text(allrun, encoding="utf-8")
    return root


def _archive_tree(source: Path, archive: Path, *, prefix: str = "") -> Path:
    with zipfile.ZipFile(archive, "w") as zip_file:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                name = path.relative_to(source).as_posix()
                zip_file.write(path, f"{prefix}{name}")
    return archive


def _nonempty_output(root: Path) -> tuple[Path, Path]:
    output = prepare_output_directory(root / "output", overwrite=False)
    # Exercise the dangerous branch: an owned output is the only kind that
    # ``--overwrite_output`` is allowed to clear.  Invalid input must still
    # leave its existing contents intact.
    sentinel = output / "do-not-delete.txt"
    sentinel.write_text("important prior result", encoding="utf-8")
    return output, sentinel


@pytest.mark.parametrize("link_kind", ["file", "directory"])
def test_invalid_symlink_source_does_not_clear_existing_output(
    tmp_path: Path,
    link_kind: str,
) -> None:
    source = _write_foundation_case(tmp_path / "source")
    external = tmp_path / "external"
    if link_kind == "file":
        external.write_text("outside", encoding="utf-8")
    else:
        external.mkdir()
    (source / "constant" / "untrusted-link").symlink_to(
        external, target_is_directory=link_kind == "directory"
    )
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="[Ss]ymbolic links"):
        import_case(source, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"
    assert external.exists()


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.txt",
        r"..\escape.txt",
        "/absolute/escape.txt",
        r"\absolute\escape.txt",
        r"C:\absolute\escape.txt",
        "C:/absolute/escape.txt",
    ],
)
def test_unsafe_zip_member_is_rejected_before_existing_output_is_cleared(
    tmp_path: Path,
    member_name: str,
) -> None:
    source = _write_foundation_case(tmp_path / "source")
    archive = _archive_tree(source, tmp_path / "case.zip")
    with zipfile.ZipFile(archive, "a") as zip_file:
        zip_file.writestr(member_name, "must never be extracted")
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="ZIP entry"):
        import_case(archive, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"
    assert not (tmp_path / "escape.txt").exists()


def test_zip_symlink_is_rejected_before_existing_output_is_cleared(tmp_path: Path) -> None:
    source = _write_foundation_case(tmp_path / "source")
    archive = _archive_tree(source, tmp_path / "case.zip")
    link_info = zipfile.ZipInfo("constant/untrusted-link")
    link_info.create_system = 3
    link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "a") as zip_file:
        zip_file.writestr(link_info, "../../outside")
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="ZIP symbolic links"):
        import_case(archive, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"


def test_archive_member_count_limit_is_checked_before_existing_output_is_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "too-many-members.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("one", "1")
        zip_file.writestr("two", "2")
    monkeypatch.setattr(case_import_source, "MAX_ARCHIVE_FILES", 1)
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="ZIP contains 2 entries"):
        import_case(archive, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"


def test_archive_uncompressed_byte_limit_is_checked_before_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "too-large.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("payload", "12")
    monkeypatch.setattr(case_import_source, "MAX_ARCHIVE_BYTES", 1)
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="uncompressed size exceeds"):
        import_case(archive, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"


def test_directory_file_limit_is_checked_before_existing_output_is_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_foundation_case(tmp_path / "source")
    monkeypatch.setattr(case_import_source, "MAX_ARCHIVE_FILES", 1)
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="import safety limit"):
        import_case(source, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"


def test_reserved_foamagent_directory_is_rejected_before_output_mutation(tmp_path: Path) -> None:
    source = _write_foundation_case(tmp_path / "source")
    (source / ".foamagent").mkdir()
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="reserved .foamagent"):
        import_case(source, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"


def test_non_case_directory_does_not_clear_existing_owned_output(tmp_path: Path) -> None:
    source = tmp_path / "not-a-case"
    source.mkdir()
    (source / "notes.txt").write_text("not an OpenFOAM case", encoding="utf-8")
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="No system/controlDict"):
        import_case(source, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"


@pytest.mark.parametrize("subdir", ["../source", "/tmp", r"..\\source"])
def test_invalid_case_subdir_does_not_clear_existing_output(
    tmp_path: Path,
    subdir: str,
) -> None:
    source = _write_foundation_case(tmp_path / "source")
    output, sentinel = _nonempty_output(tmp_path)

    with pytest.raises(CaseImportError, match="case_subdir"):
        import_case(source, output, case_subdir=subdir, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "important prior result"


def test_output_inside_source_is_rejected_without_touching_source(tmp_path: Path) -> None:
    source = _write_foundation_case(tmp_path / "source")
    marker = source / "keep-me.txt"
    marker.write_text("case input", encoding="utf-8")

    with pytest.raises(CaseImportError, match="inside the source"):
        import_case(source, source / "foamagent-output", overwrite=True)

    assert marker.read_text(encoding="utf-8") == "case input"


def test_output_containing_directory_source_is_rejected_before_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source = _write_foundation_case(output / "source")
    marker = source / "keep-me.txt"
    marker.write_text("case input", encoding="utf-8")
    (output / "prior-result.txt").write_text("prior result", encoding="utf-8")

    with pytest.raises(CaseImportError):
        import_case(source, output, overwrite=True)

    assert marker.read_text(encoding="utf-8") == "case input"
    assert (output / "prior-result.txt").read_text(encoding="utf-8") == "prior result"


def test_output_containing_zip_source_is_rejected_before_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    source = _write_foundation_case(tmp_path / "source")
    archive = _archive_tree(source, output / "case.zip")
    (output / "prior-result.txt").write_text("prior result", encoding="utf-8")

    with pytest.raises(CaseImportError):
        import_case(archive, output, overwrite=True)

    assert archive.is_file()
    assert (output / "prior-result.txt").read_text(encoding="utf-8") == "prior result"


def test_overwrite_unlinks_output_symlink_without_deleting_its_target(tmp_path: Path) -> None:
    source = _write_foundation_case(tmp_path / "source")
    output = tmp_path / "output"
    prepare_output_directory(output, overwrite=False)
    external = tmp_path / "external-result"
    external.mkdir()
    target_file = external / "preserve.txt"
    target_file.write_text("outside output", encoding="utf-8")
    (output / "external-link").symlink_to(external, target_is_directory=True)

    import_case(source, output, overwrite=True)

    assert target_file.read_text(encoding="utf-8") == "outside output"
    assert not (output / "external-link").exists()


def test_output_path_symlink_is_rejected_without_touching_its_target(tmp_path: Path) -> None:
    source = _write_foundation_case(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "preserve.txt"
    sentinel.write_text("outside output", encoding="utf-8")
    output_link = tmp_path / "output-link"
    output_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(CaseImportError, match="symlink"):
        import_case(source, output_link)

    assert sentinel.read_text(encoding="utf-8") == "outside output"


def test_output_path_through_symlinked_parent_is_rejected_without_touching_target(
    tmp_path: Path,
) -> None:
    source = _write_foundation_case(tmp_path / "source")
    target_parent = tmp_path / "real-parent"
    target_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(target_parent, target_is_directory=True)
    output = linked_parent / "output"

    with pytest.raises(CaseImportError, match="symlink"):
        import_case(source, output)

    assert not (target_parent / "output").exists()


def test_initial_execution_makes_read_only_work_copy_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first controlled run must work even when the uploaded case is read-only."""
    source = _write_foundation_case(tmp_path / "source")
    source.chmod(0o555)
    try:
        manifest = import_case(source, tmp_path / "output")
    finally:
        # Let pytest remove the input fixture after the import has preserved
        # its source permissions in the work copy.
        source.chmod(0o755)

    def successful_controlled_run(work_dir: Path, *_args: object, **_kwargs: object) -> list[object]:
        (Path(work_dir) / ".foamagent").mkdir()
        return []

    monkeypatch.setattr(case_import, "execute_imported_case", successful_controlled_run)

    result = _run_imported_manifest(manifest, timeout=1, max_repairs=0)

    assert result.status == "success"
    assert (Path(manifest.output_root) / "work").stat().st_mode & stat.S_IWUSR
    assert not (
        (Path(manifest.output_root) / "original" / "system").stat().st_mode
        & stat.S_IWUSR
    )


def test_existing_poly_mesh_without_allrun_generates_check_then_solver(tmp_path: Path) -> None:
    source = _write_foundation_case(
        tmp_path / "source", with_block_mesh=False, with_poly_mesh=True
    )

    manifest = import_case(source, tmp_path / "output")

    assert [step.command for step in manifest.execution_plan] == ["checkMesh", "icoFoam"]
    assert all(step.origin == "generated_missing_allrun" for step in manifest.execution_plan)


def test_missing_start_time_directory_blocks_synthesised_plan(tmp_path: Path) -> None:
    """A startTime restart needs its matching field directory before execution."""
    source = _write_foundation_case(tmp_path / "source")
    control_dict = source / "system" / "controlDict"
    control_dict.write_text(
        control_dict.read_text(encoding="utf-8").replace(
            "startTime 0;", "startFrom startTime;\nstartTime 0.5;"
        ),
        encoding="utf-8",
    )
    (source / "0").mkdir()

    manifest = import_case(source, tmp_path / "output")

    assert not manifest.supported
    assert manifest.blocking_issues


def test_mesh_gate_is_inserted_after_last_mesh_operation_and_before_solver() -> None:
    steps = [
        ExecutionStep("blockMesh"),
        ExecutionStep("snappyHexMesh"),
        ExecutionStep("icoFoam"),
    ]

    repaired = ensure_mesh_check(steps, "icoFoam")

    assert [step.command for step in repaired] == [
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
        "icoFoam",
    ]
    assert repaired[2].origin == "foamagent_mesh_gate"


def test_mesh_gate_does_not_accept_checkmesh_before_mesh_generation() -> None:
    steps = [
        ExecutionStep("checkMesh"),
        ExecutionStep("blockMesh"),
        ExecutionStep("icoFoam"),
    ]

    repaired = ensure_mesh_check(steps, "icoFoam")

    assert [step.command for step in repaired] == ["blockMesh", "checkMesh", "icoFoam"]


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        ([ExecutionStep("blockMesh")], ["blockMesh", "checkMesh"]),
        (
            [ExecutionStep("blockMesh"), ExecutionStep("checkMesh")],
            ["blockMesh", "checkMesh"],
        ),
    ],
)
def test_mesh_only_application_has_one_final_mesh_check(
    steps: list[ExecutionStep],
    expected: list[str],
) -> None:
    """Mesh tutorials legitimately declare blockMesh as their application."""
    repaired = ensure_mesh_check(steps, "blockMesh")

    assert [step.command for step in repaired] == expected


def test_checkmesh_application_is_not_duplicated_without_mesh_generation() -> None:
    repaired = ensure_mesh_check([ExecutionStep("checkMesh")], "checkMesh")

    assert [step.command for step in repaired] == ["checkMesh"]


def test_comment_and_readme_dependency_keywords_do_not_block_case_import(tmp_path: Path) -> None:
    source = _write_foundation_case(tmp_path / "source")
    (source / "README.md").write_text(
        "This document mentions coded function objects and #include examples.\n",
        encoding="utf-8",
    )
    control_dict = source / "system" / "controlDict"
    control_dict.write_text(
        control_dict.read_text(encoding="utf-8")
        + "// #include \"local-overrides\"\n/* coded codeExecute libs (\"not-real.so\") */\n",
        encoding="utf-8",
    )

    manifest = import_case(source, tmp_path / "output")

    assert manifest.supported
    assert manifest.blocking_issues == []
    assert manifest.detected_libraries == []


def test_allrun_parser_accepts_standard_multiline_foundation_syntax() -> None:
    steps = parse_allrun(
        """#!/bin/sh
cd ${0%/*} || exit 1
. $WM_PROJECT_DIR/bin/tools/RunFunctions
application=$(getApplication)
runApplication blockMesh \\
    -dict system/blockMeshDict
runParallel $application -parallel
""",
        "icoFoam",
    )

    assert [(step.command, step.args, step.parallel) for step in steps] == [
        ("blockMesh", ("-dict", "system/blockMeshDict"), False),
        ("icoFoam", ("-parallel",), True),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "runApplication icoFoam; touch pwned",
        "runApplication icoFoam > output",
        "runApplication icoFoam $(id)",
        "runApplication icoFoam -case ../other-case",
        "runApplication icoFoam -case=/tmp/other-case",
        "runApplication icoFoam $WM_PROJECT_DIR",
        "runParallel blockMesh",
        "rm -rf .",
    ],
)
def test_allrun_parser_rejects_shell_escape_and_unsafe_execution(line: str) -> None:
    with pytest.raises(CaseImportError):
        parse_allrun(line + "\n", "icoFoam")


def test_allrun_parser_rejects_unfinished_continuation() -> None:
    with pytest.raises(CaseImportError, match="unfinished line continuation"):
        parse_allrun("runApplication icoFoam " + chr(92), "icoFoam")


def test_controlled_allrun_stops_solver_when_checkmesh_reports_failed_checks(
    tmp_path: Path,
) -> None:
    """A zero exit from checkMesh is insufficient without a semantic Mesh OK gate."""
    case_dir = tmp_path / "case"
    controlled_dir = case_dir / ".foamagent"
    controlled_dir.mkdir(parents=True)
    fake_foam = tmp_path / "fake-openfoam"
    app_bin = fake_foam / "appbin"
    run_functions = fake_foam / "bin" / "tools" / "RunFunctions"
    app_bin.mkdir(parents=True)
    run_functions.parent.mkdir(parents=True)
    run_functions.write_text(
        """runApplication() {
    \"$@\" > \"log.$1\" 2>&1
}
runParallel() {
    runApplication \"$@\"
}
""",
        encoding="utf-8",
    )
    (app_bin / "checkMesh").write_text(
        "#!/bin/sh\nprintf 'Failed 1 mesh checks.\\n'\nexit 0\n",
        encoding="utf-8",
    )
    (app_bin / "icoFoam").write_text(
        "#!/bin/sh\ntouch solver-was-run\nprintf 'End\\n'\n",
        encoding="utf-8",
    )
    for command in (app_bin / "checkMesh", app_bin / "icoFoam"):
        command.chmod(command.stat().st_mode | stat.S_IXUSR)

    script = controlled_dir / "Allrun.controlled"
    script_text = render_controlled_allrun(
        [ExecutionStep("checkMesh"), ExecutionStep("icoFoam")]
    )
    script.write_text(script_text, encoding="utf-8")
    environment = {
        **os.environ,
        "WM_PROJECT_DIR": str(fake_foam),
        "WM_PROJECT_VERSION": "10",
        "FOAM_APPBIN": str(app_bin),
        "PATH": f"{app_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }

    result = subprocess.run(
        ["/bin/sh", str(script)],
        cwd=case_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert "foamagent_require_mesh_ok()" in script_text
    assert "foamagent_require_mesh_ok || exit $?" in script_text
    assert result.returncode == 65
    assert "refusing to start a solver" in result.stderr
    assert "Failed 1 mesh checks." in (case_dir / "log.checkMesh").read_text(
        encoding="utf-8"
    )
    assert not (case_dir / "solver-was-run").exists()


def test_safe_repairs_are_gated_by_error_type_and_leave_numbers_unchanged(
    tmp_path: Path,
) -> None:
    case = _write_foundation_case(tmp_path / "case")
    control_dict = case / "system" / "controlDict"
    malformed = control_dict.read_text(encoding="utf-8").replace(
        "object controlDict;", "object wrongName;"
    )
    control_dict.write_text(malformed, encoding="utf-8")
    before = numeric_snapshot(case)

    repairs = apply_safe_repairs(case, [{"error_content": "segmentation fault"}])

    assert repairs == []
    assert control_dict.read_text(encoding="utf-8") == malformed
    assert numeric_snapshot(case) == before


def test_missing_non_numeric_semicolon_repair_preserves_whole_case_numeric_snapshot(
    tmp_path: Path,
) -> None:
    case = _write_foundation_case(tmp_path / "case")
    control_dict = case / "system" / "controlDict"
    control_dict.write_text(
        control_dict.read_text(encoding="utf-8").replace(
            "application icoFoam;", "application icoFoam // keep this comment"
        ),
        encoding="utf-8",
    )
    before = numeric_snapshot(case)

    repairs = apply_safe_repairs(case, [{"error_content": "Expected ';'"}])

    assert any(repair["status"] == "applied" for repair in repairs)
    assert "application icoFoam; // keep this comment" in control_dict.read_text(encoding="utf-8")
    assert numeric_snapshot(case) == before


def test_object_header_repair_is_rejected_when_it_would_change_a_numeric_token(
    tmp_path: Path,
) -> None:
    case = _write_foundation_case(tmp_path / "case")
    control_dict = case / "system" / "controlDict"
    control_dict.write_text(
        control_dict.read_text(encoding="utf-8").replace(
            "object controlDict;", "object 42;"
        ),
        encoding="utf-8",
    )
    before = numeric_snapshot(case)

    repairs = apply_safe_repairs(case, [{"error_content": "FoamFile object mismatch"}])

    assert repairs == [
        {
            "file": "system/controlDict",
            "status": "rejected",
            "reason": "repair would alter numeric tokens or their line bindings",
        }
    ]
    assert "object 42;" in control_dict.read_text(encoding="utf-8")
    assert numeric_snapshot(case) == before


def test_object_header_repair_targets_the_real_foamfile_header_not_a_comment(
    tmp_path: Path,
) -> None:
    case = _write_foundation_case(tmp_path / "case")
    control_dict = case / "system" / "controlDict"
    control_dict.write_text(
        "// object misleadingComment;\n"
        + control_dict.read_text(encoding="utf-8").replace(
            "object controlDict;", "object wrongName;"
        ),
        encoding="utf-8",
    )

    repairs = apply_safe_repairs(case, [{"error_content": "FoamFile object mismatch"}])

    repaired = control_dict.read_text(encoding="utf-8")
    assert any(repair["status"] == "applied" for repair in repairs)
    assert "// object misleadingComment;" in repaired
    assert "object controlDict;" in repaired
    assert "object wrongName;" not in repaired


def test_repeated_repairable_error_stops_after_one_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_foundation_case(tmp_path / "source")
    control_dict = source / "system" / "controlDict"
    control_dict.write_text(
        control_dict.read_text(encoding="utf-8").replace(
            "object controlDict;", "object wrongName;"
        ),
        encoding="utf-8",
    )
    manifest = import_case(source, tmp_path / "output")
    calls = 0

    def same_error(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        return [{"error_content": "FoamFile object mismatch"}]

    monkeypatch.setattr(case_import, "execute_imported_case", same_error)
    result = _run_imported_manifest(manifest, timeout=1, max_repairs=5)

    assert result.status == "blocked"
    assert calls == 2
    assert result.attempts[-1]["terminal_reason"] == "repeated_error_fingerprint"
    assert "object controlDict;" in (
        tmp_path / "output" / "work" / "system" / "controlDict"
    ).read_text(encoding="utf-8")
    assert "object wrongName;" in (
        tmp_path / "output" / "original" / "system" / "controlDict"
    ).read_text(encoding="utf-8")
