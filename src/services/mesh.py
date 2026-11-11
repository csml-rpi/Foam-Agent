"""Native mesh backends: one attempt per main-graph mesh execution."""
import json
import os
import re
import shutil
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from utils import run_openfoam_utility, run_owned_process, save_file, reject_symbolic_links
from . import get_llm_service
from .run_hpc import HpcWorkflowStop
from .visualization import ensure_foam_file
from .execution_logs import check_execution_outputs
from .case_paths import safe_case_path
from .case_inputs import read_case_inputs


_FAILED_MESH_CHECKS_RE = re.compile(r'Failed [1-9][0-9]* mesh checks')
_MESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")


def _missing_mesh_files(directory):
    return [name for name in _MESH_FILES
            if not ((directory / name).is_file() or (directory / (name + ".gz")).is_file())]


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


def prepare_custom_mesh(custom_mesh_path, case_dir, *, retry=False, source_written=False,
                        protected_inputs=()):
    conflicts = check_execution_outputs(['constant/polyMesh'], protected_inputs)
    if conflicts:
        return conflicts
    mesh_in_case_dir = safe_case_path(case_dir, '', 'geometry.msh')
    origin_file = safe_case_path(case_dir, '.foamagent', 'mesh-source.json')
    origin = dict(source=str(Path(custom_mesh_path).expanduser().resolve()), backend='custom_mesh')
    previous = json.loads(origin_file.read_text()) if origin_file.is_file() else None
    if source_written or not retry or previous != origin or not mesh_in_case_dir.is_file():
        if not (mesh_in_case_dir.exists() and os.path.samefile(custom_mesh_path, mesh_in_case_dir)):
            conflicts = check_execution_outputs(['geometry.msh'], protected_inputs)
            if conflicts:
                return conflicts
            mesh_in_case_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(custom_mesh_path, mesh_in_case_dir)
    origin_file.parent.mkdir(parents=True, exist_ok=True)
    origin_file.write_text(json.dumps(origin))
    return []


def copy_custom_mesh(custom_mesh_path: str, case_dir: str, *, timeout: int,
                     openfoam_target: str = '', utility_runner=None, retry: bool = False,
                     source_written: bool = False, protected_inputs=()) -> Dict[str, Any]:
    """Stage the selected .msh, convert and check it once; preserve repaired working copies."""
    execution = []
    if not custom_mesh_path:
        return _mesh_failure('No custom mesh path provided', execution)
    if not os.path.exists(custom_mesh_path):
        return _mesh_failure(f'Custom mesh not found: {custom_mesh_path}', execution)
    mesh_in_case_dir = Path(case_dir)/'geometry.msh'
    try:
        reject_symbolic_links(Path(case_dir) / 'constant')
        reject_symbolic_links(Path(case_dir) / 'system')
        conflicts = prepare_custom_mesh(custom_mesh_path, case_dir, retry=retry,
                                        source_written=source_written, protected_inputs=protected_inputs)
        if conflicts:
            result = _mesh_failure('Custom mesh outputs conflict with declared startup inputs', execution)
            result['error_logs'] = conflicts
            return result
        (Path(case_dir)/'constant').mkdir(exist_ok=True)
        (Path(case_dir)/'system').mkdir(exist_ok=True)
        _convert_mesh(case_dir, timeout, openfoam_target, utility_runner, execution)
    except HpcWorkflowStop:
        raise
    except ValueError as error:
        return _mesh_failure(f'Invalid mesh-source evidence: {error}', execution)
    except subprocess.CalledProcessError as error:
        return _mesh_failure(f'Custom mesh conversion/check failed: {error.stderr or error.stdout or error}', execution)
    except subprocess.TimeoutExpired:
        return _mesh_failure(f'Custom mesh conversion/check timed out after {timeout}s', execution)
    except (OSError, RuntimeError) as error:
        return _mesh_failure(f'Custom mesh conversion/check failed: {error}', execution)
    if not (Path(case_dir)/'constant/polyMesh').is_dir():
        return _mesh_failure('polyMesh directory not created', execution)
    ensure_foam_file(case_dir)
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


def prepare_gmsh_script(user_requirement, case_dir, *, script_prepared, llm_service=None,
                        openfoam_target='', repair_feedback=''):
    python_file = safe_case_path(case_dir, '', 'generate_mesh.py')
    changes = {'updated_files': [], 'file_diffs': []}
    if not script_prepared:
        existing = python_file.read_text() if python_file.is_file() else ''
        inputs, omitted = read_case_inputs(case_dir, selected_paths=['generate_mesh.py'])
        existing_context = next((file.content for file in inputs.list_foamfile
                                 if file.file_name == python_file.name), '')
        client = get_llm_service(llm_service)
        response = client.invoke(
            f'<user_requirements>{user_requirement}</user_requirements>\n'
            f'<openfoam_target>{openfoam_target}</openfoam_target>\n'
            f'<existing_python_code>{existing_context}</existing_python_code>\n'
            f'<omitted_files>{json.dumps(omitted)}</omitted_files>\n'
            f'<repair_feedback>{repair_feedback}</repair_feedback>',
            GMSH_PYTHON_SYSTEM_PROMPT, pydantic_obj=GMSHPythonCode)
        if not response.python_code.strip():
            raise RuntimeError('Gmsh code generation returned an empty script.')
        save_file(str(python_file), response.python_code)
        if existing != response.python_code:
            changes['updated_files'].append('generate_mesh.py')
            changes['file_diffs'].append(''.join(
                line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                for line in unified_diff(
                    existing.splitlines(keepends=True), response.python_code.splitlines(keepends=True),
                    fromfile='before/generate_mesh.py', tofile='after/generate_mesh.py')))
    if not python_file.is_file():
        raise RuntimeError('The declared prepared Gmsh script is missing.')
    return changes


def handle_gmsh_mesh(user_requirement: str, case_dir: str, *, timeout: int, script_prepared: bool,
                     llm_service: Optional[Any] = None, openfoam_target: str = '', utility_runner=None,
                     repair_feedback: str = '', python_runner=None, protected_inputs=()) -> Dict[str, Any]:
    """Generate or execute the explicitly written script, then convert/check once."""
    case_dir = os.path.abspath(case_dir)
    python_file = Path(case_dir)/'generate_mesh.py'
    msh_file = Path(case_dir)/'geometry.msh'
    execution = []
    conflicts = check_execution_outputs(['geometry.msh', 'constant/polyMesh'], protected_inputs)
    if conflicts:
        result = _mesh_failure('Gmsh outputs conflict with declared startup inputs', execution)
        result['error_logs'] = conflicts
        return result
    try:
        reject_symbolic_links(Path(case_dir) / 'constant')
        reject_symbolic_links(Path(case_dir) / 'system')
        os.makedirs(case_dir, exist_ok=True)
        prepare_gmsh_script(user_requirement, case_dir, script_prepared=script_prepared,
                            llm_service=llm_service, openfoam_target=openfoam_target,
                            repair_feedback=repair_feedback)
        _clear_gmsh_attempt_outputs(case_dir)
        _run_mesh_command(execution, python_runner or run_owned_process, [sys.executable, str(python_file)],
                          cwd=case_dir, check=True, text=True, timeout=timeout)
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
    except (OSError, RuntimeError, ValueError) as error:
        return _mesh_failure(f'Mesh setup failed: {error}', execution)
    if not (Path(case_dir)/'constant/polyMesh').is_dir():
        return _mesh_failure('polyMesh directory not created', execution)
    ensure_foam_file(case_dir)
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
