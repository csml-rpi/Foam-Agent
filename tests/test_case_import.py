"""Case import, target discovery, and imported-case repair."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import foambench_main
import pytest
from services.case_import import import_case
from services.input_writer import rewrite_files
from utils import FoamPydantic, FoamfilePydantic, read_case_foamfiles, scan_case_directory


@pytest.mark.parametrize('invalid', ['zip', 'case_selection'])
def test_invalid_import_preserves_previous_managed_output(tmp_path, invalid):
    from services.case_import import import_case
    from services.output_safety import prepare_output_directory
    from models import CaseImportError
    output = prepare_output_directory(tmp_path / 'output', overwrite=False)
    previous = output / 'work/0/U'
    previous.parent.mkdir(parents=True)
    previous.write_text('previous valid field')
    archive = tmp_path / 'source.zip'
    if invalid == 'zip':
        archive.write_text('broken archive')
    else:
        with zipfile.ZipFile(archive, 'w') as zipped:
            for case in ('a', 'b'):
                zipped.writestr(f'{case}/system/controlDict', 'application icoFoam;')
    with pytest.raises(CaseImportError):
        import_case(archive, output, overwrite=True, openfoam_target='foundation-v10')
    assert previous.read_text() == 'previous valid field'


@pytest.fixture(autouse=True)
def _restore_writable_tmp_tree(tmp_path: Path):
    """Allow pytest to remove immutable original/ trees after each test."""
    yield
    paths = [tmp_path, *tmp_path.rglob("*")]
    for path in paths:
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode | 0o700)
        except OSError:
            pass


_NATIVE_HEADERS = {
    'foundation-v10': r'''/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
''',
    'esi-v2006': r'''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
''',
}


def _make_case(root: Path, target: str = "foundation-v10") -> Path:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "0").mkdir()
    header = _NATIVE_HEADERS[target]
    (root / "system" / "controlDict").write_text(
        header + "application icoFoam;\nstartTime 0;\nendTime 1;\n",
        encoding="utf-8",
    )
    (root / "system" / "fvSchemes").write_text("ddtSchemes {}\n", encoding="utf-8")
    (root / "system" / "fvSolution").write_text("solvers {}\n", encoding="utf-8")
    (root / "constant" / "polyMesh").mkdir()
    (root / "0" / "U").write_text("internalField uniform (0 0 0);\n", encoding="utf-8")
    (root / "Allrun").write_text("#!/bin/sh\nicoFoam\n", encoding="utf-8")
    return root


@pytest.mark.parametrize('archive_input,relative', [
    (False, 'constant/polyMesh/points'),
    (True, '0/U'),
])
def test_incomplete_case_import_preserves_the_supplied_root(tmp_path, archive_input, relative):
    source = tmp_path / 'supplied'
    supplied = source / relative
    supplied.parent.mkdir(parents=True)
    content = b'Supplied case input\n'
    supplied.write_bytes(content)
    if archive_input:
        archive = tmp_path / 'case.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            output.write(supplied, relative)
        source = archive
    output = tmp_path / 'imported'

    manifest = import_case(source, output, openfoam_target='foundation-v10')

    assert manifest.case_root == '.'
    for name in ('original', 'work'):
        files = {path.relative_to(output / name).as_posix(): path.read_bytes()
                 for path in (output / name).rglob('*') if path.is_file()}
        assert files == {relative: content}
    assert supplied.read_bytes() == content


class _FakeLLM:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def invoke(self, user_prompt, system_prompt, pydantic_obj=None):
        self.calls.append((user_prompt, system_prompt, pydantic_obj))
        return self.response


@pytest.mark.parametrize("target", ["foundation-v10", "esi-v2006"])
def test_import_materialises_original_and_work_and_detects_target(tmp_path: Path, target: str) -> None:
    source = _make_case(tmp_path / "source", target)
    output = tmp_path / "task"

    manifest = import_case(source, output)

    assert manifest.platform == target
    assert manifest.application == "icoFoam"
    assert (output / "original" / "system" / "controlDict").is_file()
    assert (output / "work" / "system" / "controlDict").is_file()
    assert not os.access(output / "original" / "system" / "controlDict", os.W_OK)
    assert (output / "report" / "case_context.json").is_file()


def test_modify_existing_applies_only_planned_target_files(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    control_before = (case / "system" / "controlDict").read_text(encoding="utf-8")
    solution_before = (case / "system" / "fvSolution").read_text(encoding="utf-8")
    response = FoamPydantic(
        list_foamfile=[
            FoamfilePydantic(
                folder_name="system",
                file_name="controlDict",
                content=control_before.replace("endTime 1", "endTime 5"),
            ),
            FoamfilePydantic(
                folder_name="system",
                file_name="fvSolution",
                content="solvers { changed yes; }\n",
            ),
            FoamfilePydantic(folder_name="0", file_name="p", content="internalField uniform 0;\n"),
        ]
    )

    result = rewrite_files(
        case_dir=str(case),
        error_logs=[],
        review_analysis="Only update the requested end time.",
        rewrite_plan={
            "target_files": [
                {"file": "system/controlDict", "changes": "Set endTime to 5", "stage": "solver"}
            ]
        },
        user_requirement="Set endTime to 5.",
        foamfiles=read_case_foamfiles(str(case), scan_case_directory(str(case))),
        dir_structure=scan_case_directory(str(case)),
        llm_service=_FakeLLM(response),
    )

    assert "endTime 5" in (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert (case / "system" / "fvSolution").read_text(encoding="utf-8") == solution_before
    assert not (case / "0" / "p").exists()
    assert result["updated_files"] == ["system/controlDict"]


def test_cli_accepts_case_and_prompt_together(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "foambench_main.py",
            "--case_path",
            "./case",
            "--prompt_path",
            "./requirements.txt",
            "--openfoam_target",
            "esi-v2006",
        ],
    )

    args = foambench_main.parse_args()

    assert args.case_path == "./case"
    assert args.prompt_path == "./requirements.txt"
    assert args.openfoam_target == "esi-v2006"


@pytest.mark.parametrize('archive_input', [False, True])
def test_helper_is_executable_after_directory_or_zip_import(tmp_path, archive_input):
    from services.case_import import import_case

    source = tmp_path / 'source'
    (source / 'system').mkdir(parents=True)
    (source / 'system/controlDict').write_text('application icoFoam;\n')
    helper = source / 'prepare-geometry'
    helper.write_text('#!/bin/sh\nprintf "geometry prepared\\n"\n')
    helper.chmod(0o755)
    if archive_input:
        archive = tmp_path / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            for path in source.rglob('*'):
                output.write(path, path.relative_to(source))
        source = archive
    destination = tmp_path / 'imported'
    import_case(source, destination)
    original, work = destination / 'original/prepare-geometry', destination / 'work/prepare-geometry'
    assert not original.stat().st_mode & 0o222
    assert work.stat().st_mode & stat.S_IWUSR
    result = subprocess.run([str(work)], check=True, capture_output=True, text=True)
    assert result.stdout == 'geometry prepared\n'
    assert work.stat().st_mode & 0o111 == original.stat().st_mode & 0o111 == 0o111
