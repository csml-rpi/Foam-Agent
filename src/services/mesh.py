"""Native mesh backends: one attempt per main-graph mesh execution."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from utils import run_openfoam_utility, save_file
from . import global_llm_service
from .run_hpc import HpcWorkflowStop


_FAILED_MESH_CHECKS_RE = re.compile(r'Failed [1-9][0-9]* mesh checks')


def _mesh_failure(message, execution):
    return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None,
            "error_logs": [message], "native_execution": execution}


def _run_mesh_command(execution, runner, argv, **kwargs):
    """Retain actual native outcomes, including failed or timed-out commands."""
    entry = dict(argv=argv, timeout=kwargs['timeout'], returncode=None, stdout='', stderr='')
    execution.append(entry)
    try:
        result = runner(argv, **kwargs)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        entry.update(error_type=type(error).__name__,
                     returncode=error.returncode if isinstance(error, subprocess.CalledProcessError) else None,
                     stdout=str(getattr(error, 'stdout', '') or ''),
                     stderr=str(getattr(error, 'stderr', '') or ''))
        raise
    entry.update(returncode=result.returncode, stdout=result.stdout or '', stderr=result.stderr or '')
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, argv, output=result.stdout, stderr=result.stderr)
    return result


def _convert_mesh(case_dir, timeout, openfoam_target, utility_runner, execution):
    runner = utility_runner or run_openfoam_utility
    _run_mesh_command(execution, runner, ['gmshToFoam', 'geometry.msh'],
                      working_dir=case_dir, timeout=timeout, openfoam_target=openfoam_target)
    checked = _run_mesh_command(execution, runner, ['checkMesh'], working_dir=case_dir,
                                timeout=timeout, openfoam_target=openfoam_target)
    if _FAILED_MESH_CHECKS_RE.search(checked.stdout or ''):
        raise RuntimeError('checkMesh reported failed mesh checks.\n' + checked.stdout)


def copy_custom_mesh(custom_mesh_path: str, case_dir: str, *, timeout: int,
                     openfoam_target: str = '', utility_runner=None, retry: bool = False,
                     source_written: bool = False) -> Dict[str, Any]:
    """Stage the selected .msh, convert and check it once; preserve repaired working copies."""
    execution = []
    if not custom_mesh_path:
        return _mesh_failure('No custom mesh path provided', execution)
    if not os.path.exists(custom_mesh_path):
        return _mesh_failure(f'Custom mesh not found: {custom_mesh_path}', execution)
    os.makedirs(case_dir, exist_ok=True)
    mesh_in_case_dir = Path(case_dir)/'geometry.msh'
    origin_file = Path(case_dir)/'.foamagent/mesh-source.json'
    origin = dict(source=str(Path(custom_mesh_path).expanduser().resolve()), backend='custom_mesh')
    try:
        previous = json.loads(origin_file.read_text()) if origin_file.is_file() else None
    except ValueError as error:
        return _mesh_failure(f'Invalid mesh-source evidence: {error}', execution)
    try:
        # A declared source write must reach conversion; other retries retain
        # any repairs made directly to the working geometry.msh.
        if source_written or not retry or previous != origin or not mesh_in_case_dir.is_file():
            if not (mesh_in_case_dir.exists() and os.path.samefile(custom_mesh_path, mesh_in_case_dir)):
                shutil.copy2(custom_mesh_path, mesh_in_case_dir)
        origin_file.parent.mkdir(parents=True, exist_ok=True)
        origin_file.write_text(json.dumps(origin))
        (Path(case_dir)/'constant').mkdir(exist_ok=True)
        (Path(case_dir)/'system').mkdir(exist_ok=True)
        _convert_mesh(case_dir, timeout, openfoam_target, utility_runner, execution)
    except HpcWorkflowStop:
        raise
    except subprocess.CalledProcessError as error:
        return _mesh_failure(f'Custom mesh conversion/check failed: {error.stderr or error.stdout or error}', execution)
    except subprocess.TimeoutExpired:
        return _mesh_failure(f'Custom mesh conversion/check timed out after {timeout}s', execution)
    except (OSError, RuntimeError) as error:
        return _mesh_failure(f'Custom mesh conversion/check failed: {error}', execution)
    if not (Path(case_dir)/'constant/polyMesh').is_dir():
        return _mesh_failure('polyMesh directory not created', execution)
    (Path(case_dir)/(Path(case_dir).name+'.foam')).touch()
    return dict(mesh_info=dict(mesh_file_path=str(mesh_in_case_dir), mesh_file_type='gmsh',
                              mesh_description='Selected mesh converted and checked',
                              requires_blockmesh_removal=True, check_mesh_passed=True),
                mesh_commands=[], custom_mesh_used=True, error_logs=[], native_execution=execution)


GMSH_PYTHON_SYSTEM_PROMPT = (
    'Generate a complete Gmsh Python program from the supplied approved mesh plan, '
    'user requirements, actual input files and target-specific tutorial guidance. '
    'Preserve existing script content except for requested changes and diagnosed repairs. '
    'Define the requested geometry and physical groups using their supplied evidence. '
    'Generate a volume mesh suitable for OpenFOAM conversion, write geometry.msh using '
    'MSH version 2.2, and finalize Gmsh. Geometric mesh work belongs here; physical '
    'field initialization and solving belong to the solver stage. '
    'Return code using the supplied structured schema. Native failures return to the '
    'main Mesh Reviewer and Meshing Planner; do not add an internal retry loop.'
)


class GMSHPythonCode(BaseModel):
    python_code: str = Field(description='Complete Python code using Gmsh')
    mesh_type: str = Field(description='Planned mesh dimensionality')
    geometry_type: str = Field(description='Planned geometry description')


def handle_gmsh_mesh(user_requirement: str, case_dir: str, *, timeout: int, script_prepared: bool,
                     llm_service: Optional[Any] = None, openfoam_target: str = '', utility_runner=None,
                     repair_feedback: str = '', python_runner=None) -> Dict[str, Any]:
    """Generate or execute the explicitly written script, then convert/check once."""
    case_dir = os.path.abspath(case_dir)
    os.makedirs(case_dir, exist_ok=True)
    python_file = Path(case_dir)/'generate_mesh.py'
    msh_file = Path(case_dir)/'geometry.msh'
    execution = []
    try:
        if not script_prepared:
            existing = python_file.read_text() if python_file.is_file() else ''
            client = llm_service if llm_service is not None else global_llm_service
            response = client.invoke(
                f'<user_requirements>{user_requirement}</user_requirements>\n'
                f'<openfoam_target>{openfoam_target}</openfoam_target>\n'
                f'<existing_python_code>{existing}</existing_python_code>\n'
                f'<repair_feedback>{repair_feedback}</repair_feedback>',
                GMSH_PYTHON_SYSTEM_PROMPT, pydantic_obj=GMSHPythonCode)
            if not response.python_code.strip():
                return _mesh_failure('Gmsh code generation returned an empty script.', execution)
            save_file(str(python_file), response.python_code)
        if not python_file.is_file():
            return _mesh_failure('The declared prepared Gmsh script is missing.', execution)
        _clear_gmsh_attempt_outputs(case_dir)
        _run_mesh_command(execution, python_runner or subprocess.run, [sys.executable, str(python_file)],
                          cwd=case_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout)
        if not msh_file.is_file():
            return _mesh_failure('Gmsh script completed without creating geometry.msh.', execution)
        (Path(case_dir)/'constant').mkdir(exist_ok=True)
        (Path(case_dir)/'system').mkdir(exist_ok=True)
        _convert_mesh(case_dir, timeout, openfoam_target, utility_runner, execution)
    except HpcWorkflowStop:
        raise
    except subprocess.CalledProcessError as error:
        return _mesh_failure(f'Mesh command failed: {error.stderr or error.stdout or error}', execution)
    except subprocess.TimeoutExpired:
        return _mesh_failure(f'Mesh command timed out after {timeout}s.', execution)
    except (OSError, RuntimeError) as error:
        return _mesh_failure(f'Mesh setup failed: {error}', execution)
    if not (Path(case_dir)/'constant/polyMesh').is_dir():
        return _mesh_failure('polyMesh directory not created', execution)
    (Path(case_dir)/(Path(case_dir).name+'.foam')).touch()
    return dict(mesh_info=dict(mesh_file_path=str(msh_file), mesh_file_type='gmsh',
                              mesh_description='Gmsh program executed, converted and checked',
                              requires_blockmesh_removal=True, check_mesh_passed=True),
                mesh_commands=[], mesh_file_destination=str(msh_file), custom_mesh_used=True,
                error_logs=[], native_execution=execution)


def _clear_gmsh_attempt_outputs(case_dir: str) -> None:
    """Remove only artifacts regenerated by the next Gmsh attempt.

    A failed script can leave a valid-looking ``geometry.msh`` or ``polyMesh``.
    Keeping either would let a later failed retry convert or validate stale data.
    """
    msh_file = Path(case_dir) / "geometry.msh"
    if msh_file.is_symlink() or msh_file.is_file():
        msh_file.unlink()
    elif msh_file.exists():
        raise OSError(f"Expected mesh artifact is not a regular file: {msh_file}")

    poly_mesh = Path(case_dir) / "constant" / "polyMesh"
    if poly_mesh.is_symlink():
        poly_mesh.unlink()
    elif poly_mesh.exists():
        if not poly_mesh.is_dir():
            raise OSError(f"Expected polyMesh artifact is not a directory: {poly_mesh}")
        shutil.rmtree(poly_mesh)
