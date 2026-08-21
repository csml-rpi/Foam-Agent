import os
import re
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from models import RunOut
from utils import remove_numeric_folders, run_command, check_foam_errors
from .allrun_commands import (
    application_positions as allrun_application_positions,
    command_positions as allrun_command_positions,
    script_without_comments as allrun_script_without_comments,
)
from .openfoam_commands import MESH_MUTATING_COMMANDS


_MESH_GENERATION_COMMANDS = MESH_MUTATING_COMMANDS
_FOAM_POINT_RE = re.compile(
    r"\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)"
)


def _validation_error(file_name: str, message: str) -> Dict[str, str]:
    return {"file": file_name, "error_content": message}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _strip_foam_comments(content: str) -> str:
    """Remove OpenFOAM line and block comments before light syntax checks."""
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return re.sub(r"//.*$", "", content, flags=re.MULTILINE)


def _control_dict_application(case_dir: str) -> Optional[str]:
    control_dict = _read_text(Path(case_dir) / "system" / "controlDict")
    control_dict = _strip_foam_comments(control_dict)
    match = re.search(r"\bapplication\s+([^\s;]+)\s*;", control_dict)
    return match.group(1) if match else None


def _mesh_generation_positions(allrun_script: str) -> List[int]:
    positions = []
    for command in _MESH_GENERATION_COMMANDS:
        positions.extend(allrun_command_positions(allrun_script, command))
    return sorted(positions)


def validate_momentum_transport_dictionaries(case_dir: str) -> List[str]:
    """Require the Foundation transport-model selector when such a file exists.

    The check is filename- and syntax-based, not solver- or case-specific.  It
    supports both single-phase ``momentumTransport`` and phase-qualified
    ``momentumTransport.<phase>`` files.
    """
    constant_dir = Path(case_dir) / "constant"
    try:
        dictionaries = sorted(
            path for path in constant_dir.glob("momentumTransport*")
            if path.is_file()
        )
    except OSError:
        return []

    issues: List[str] = []
    for dictionary in dictionaries:
        content = _strip_foam_comments(_read_text(dictionary))
        if not re.search(r"\bsimulationType\s+[^\s;]+\s*;", content):
            issues.append(
                f"constant/{dictionary.name} is missing the required Foundation "
                "OpenFOAM simulationType entry"
            )
    return issues


def _balanced_foam_block(text: str, keyword: str, opener: str, closer: str) -> str:
    """Return a balanced OpenFOAM list/dictionary body following *keyword*.

    This intentionally handles only the balanced delimiters needed for the
    light-weight semantic checks below; OpenFOAM itself remains the authority
    for complete dictionary parsing.
    """
    match = re.search(rf"\b{re.escape(keyword)}\b", text)
    if not match:
        return ""
    start = text.find(opener, match.end())
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return ""


def _blockmesh_boundary_patches(content: str) -> List[tuple[str, str, str]]:
    """Extract top-level ``boundary`` patch names, types, and faces text."""
    boundary = _balanced_foam_block(content, "boundary", "(", ")")
    patches: List[tuple[str, str, str]] = []
    cursor = 0
    while cursor < len(boundary):
        match = re.search(r"\b([A-Za-z_][\w.-]*)\s*\{", boundary[cursor:])
        if not match:
            break
        name = match.group(1)
        opening = cursor + match.end() - 1
        depth = 0
        closing = -1
        for index in range(opening, len(boundary)):
            if boundary[index] == "{":
                depth += 1
            elif boundary[index] == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing < 0:
            break
        patch = boundary[opening + 1:closing]
        type_match = re.search(r"\btype\s+([^\s;]+)\s*;", patch)
        faces = _balanced_foam_block(patch, "faces", "(", ")")
        patches.append((name, type_match.group(1) if type_match else "", faces))
        cursor = closing + 1
    return patches


def _canonical_face_plane(
    face: List[int],
    vertices: List[tuple[float, float, float]],
) -> Optional[tuple[tuple[float, float, float], float]]:
    """Return a sign-independent plane normal and offset for one mesh face."""
    if len(face) < 3 or any(index < 0 or index >= len(vertices) for index in face):
        return None
    first, second, third = (vertices[index] for index in face[:3])
    edge_a = tuple(second[i] - first[i] for i in range(3))
    edge_b = tuple(third[i] - first[i] for i in range(3))
    normal = (
        edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
        edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
        edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
    )
    magnitude = math.sqrt(sum(component * component for component in normal))
    if magnitude == 0:
        return None
    normal = tuple(component / magnitude for component in normal)
    for component in normal:
        if abs(component) > 1e-12:
            if component < 0:
                normal = tuple(-value for value in normal)
            break
    return normal, sum(normal[i] * first[i] for i in range(3))


def validate_blockmesh_symmetry_planes(case_dir: str) -> List[str]:
    """Catch a frequent invalid ``symmetryPlane`` construction before running.

    A ``symmetryPlane`` patch represents one plane.  This check is deliberately
    geometry-only: it neither knows the case name nor rewrites generated input.
    It simply reports a violation for the normal LLM/reviewer repair loop.
    """
    content = _read_text(Path(case_dir) / "system" / "blockMeshDict")
    if not content:
        return []
    vertices_body = _balanced_foam_block(content, "vertices", "(", ")")
    vertices = [
        tuple(float(component) for component in match.groups())
        for match in _FOAM_POINT_RE.finditer(vertices_body)
    ]
    if not vertices:
        return []

    issues: List[str] = []
    face_pattern = re.compile(r"\(\s*((?:\d+\s+)*\d+)\s*\)")
    for name, patch_type, faces_body in _blockmesh_boundary_patches(content):
        if patch_type != "symmetryPlane":
            continue
        faces = [
            [int(index) for index in match.group(1).split()]
            for match in face_pattern.finditer(faces_body)
        ]
        planes = [
            _canonical_face_plane(face, vertices)
            for face in faces
        ]
        planes = [plane for plane in planes if plane is not None]
        if len(planes) < 2:
            continue
        reference_normal, reference_offset = planes[0]
        tolerance = 1e-9
        for normal, offset in planes[1:]:
            normal_alignment = sum(
                reference_normal[index] * normal[index]
                for index in range(3)
            )
            if normal_alignment < 1 - tolerance or abs(offset - reference_offset) > tolerance:
                issues.append(
                    "Preflight failed: system/blockMeshDict patch "
                    f"{name!r} declares type symmetryPlane but contains faces on "
                    "different planes. Use type symmetry or one symmetryPlane "
                    "patch per plane."
                )
                break
    return issues


def validate_openfoam_case_preflight(
    case_dir: str,
    allrun_script: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Validate generic execution contracts before launching OpenFOAM."""
    allrun_path = Path(case_dir) / "Allrun"
    script = allrun_script if allrun_script is not None else _read_text(allrun_path)
    script_without_comments = allrun_script_without_comments(script)
    errors: List[Dict[str, str]] = []

    mesh_positions = _mesh_generation_positions(script_without_comments)
    if allrun_command_positions(script_without_comments, "blockMesh"):
        errors.extend(
            _validation_error("system/blockMeshDict", message)
            for message in validate_blockmesh_symmetry_planes(case_dir)
        )
    errors.extend(
        _validation_error("constant/momentumTransport", message)
        for message in validate_momentum_transport_dictionaries(case_dir)
    )
    check_mesh_positions = allrun_command_positions(script_without_comments, "checkMesh")
    application = _control_dict_application(case_dir)
    application_positions = allrun_application_positions(script_without_comments, application)
    if mesh_positions:
        last_mesh_position = max(mesh_positions)
        valid_check_positions = [
            position for position in check_mesh_positions
            if position > last_mesh_position
        ]
        if not valid_check_positions:
            errors.append(_validation_error(
                "Allrun",
                "Preflight failed: Allrun generates or imports a mesh but does not "
                "run checkMesh after mesh generation.",
            ))

        # A few official mesh tutorials declare ``blockMesh`` itself as the
        # controlDict application.  In that case it is a mesh utility, not a
        # solver that must be preceded by checkMesh: the correct order is
        # blockMesh followed by checkMesh.  Keep the stricter gate for actual
        # solvers, which must not start until the mesh check has passed.
        if (
            valid_check_positions
            and application_positions
            and application not in _MESH_GENERATION_COMMANDS
        ):
            first_solver_position = min(application_positions)
            has_check_before_solver = any(
                last_mesh_position < position < first_solver_position
                for position in check_mesh_positions
            )
            if not has_check_before_solver:
                errors.append(_validation_error(
                    "Allrun",
                    "Preflight failed: checkMesh must run after mesh generation and "
                    f"before the configured solver {application}.",
                ))
    elif (
        check_mesh_positions
        and application_positions
        and application not in _MESH_GENERATION_COMMANDS
    ):
        # A custom mesh can be converted before Allrun is generated. In that
        # workflow there is no generator command in Allrun, but the quality
        # gate must still precede the solver.
        if min(check_mesh_positions) >= min(application_positions):
            errors.append(_validation_error(
                "Allrun",
                "Preflight failed: checkMesh must run before the configured "
                f"solver {application}.",
            ))

    return errors


def _matching_log_files(case_dir: str, prefix: str) -> List[Path]:
    case_path = Path(case_dir)
    try:
        return sorted(
            path for path in case_path.iterdir()
            if path.is_file() and path.name.startswith(prefix)
        )
    except OSError:
        return []


def validate_openfoam_case_postflight(
    case_dir: str,
    allrun_script: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Validate mesh and solver evidence after Allrun has exited."""
    allrun_path = Path(case_dir) / "Allrun"
    script = allrun_script if allrun_script is not None else _read_text(allrun_path)
    script_without_comments = allrun_script_without_comments(script)
    errors: List[Dict[str, str]] = []

    mesh_positions = _mesh_generation_positions(script_without_comments)
    check_mesh_positions = allrun_command_positions(script_without_comments, "checkMesh")
    allrun_output = _read_text(Path(case_dir) / "Allrun.out")
    if mesh_positions or check_mesh_positions:
        check_mesh_logs = _matching_log_files(case_dir, "log.checkMesh")
        mesh_evidence = [_read_text(path) for path in check_mesh_logs]
        if allrun_output:
            # Backward compatibility for hand-written Allrun scripts which
            # invoke checkMesh directly instead of through runApplication.
            mesh_evidence.append(allrun_output)
        if not mesh_evidence:
            errors.append(_validation_error(
                "log.checkMesh",
                "Postflight failed: mesh validation was requested, but "
                "neither log.checkMesh nor Allrun.out was produced.",
            ))
        elif not any(
            re.search(r"\bMesh\s+OK\b", content, re.IGNORECASE)
            for content in mesh_evidence
        ):
            errors.append(_validation_error(
                "log.checkMesh",
                "Postflight failed: checkMesh did not report 'Mesh OK'.",
            ))

    application = _control_dict_application(case_dir)
    if application and application not in _MESH_GENERATION_COMMANDS:
        solver_logs = _matching_log_files(case_dir, f"log.{application}")
        solver_evidence = [_read_text(path) for path in solver_logs]
        # Direct command scripts do not use RunFunctions and write to the
        # combined Allrun output instead.  Treat that as evidence only if the
        # script demonstrably invokes the configured application.
        if not solver_evidence and allrun_application_positions(script_without_comments, application):
            solver_evidence.append(allrun_output)
        if not solver_evidence:
            errors.append(_validation_error(
                f"log.{application}",
                "Postflight failed: the configured solver did not produce a log file.",
            ))
        elif not any(re.search(r"\bEnd\b", content) for content in solver_evidence):
            errors.append(_validation_error(
                f"log.{application}",
                "Postflight failed: solver output does not contain the OpenFOAM End marker.",
            ))

    return errors


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    """Read a run result while remaining compatible with older monkeypatches."""
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _deduplicate_errors(errors: List[Any]) -> List[Any]:
    deduplicated = []
    seen = set()
    for error in errors:
        if isinstance(error, dict):
            key = (error.get("file"), error.get("error_content"))
        else:
            key = str(error)
        if key not in seen:
            seen.add(key)
            deduplicated.append(error)
    return deduplicated


def _cleanup_run_artifacts(case_dir: str) -> None:
    """Remove only disposable solver outputs before a local retry.

    A retry must not inherit processor partitions, post-processing output, or
    numeric result times from a failed previous execution.  The ``0`` initial
    condition and all dictionaries are deliberately retained.  This routine
    is used only for generated/local cases; imported cases use their stricter
    original-to-work restore mechanism.
    """
    root = Path(case_dir)
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_symlink():
                # A case output symlink is not a valid solver artifact, but
                # unlinking the link never follows it.
                if entry.name.startswith("log"):
                    entry.unlink()
                continue
            if entry.is_file() and (
                entry.name.startswith("log")
                or entry.name in {"Allrun.out", "Allrun.err"}
            ):
                entry.unlink()
            elif entry.is_dir() and (
                re.fullmatch(r"processor\d+", entry.name)
                or entry.name in {"postProcessing", "VTK"}
            ):
                shutil.rmtree(entry)
        except OSError as exc:
            raise RuntimeError(f"Unable to clean prior run artifact {entry}: {exc}") from exc
    remove_numeric_folders(case_dir)

def run_allrun_and_collect_errors(
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1
) -> List[Any]:
    """
    Execute the Allrun script and collect any error logs from the simulation.
    
    This function runs the Allrun script in the specified case directory,
    captures the output and error streams, and parses the results to identify
    any OpenFOAM errors that occurred during execution.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case and Allrun script
        timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
    
    Returns:
        List[Any]: Structured error records found in the simulation logs.
                 Empty list indicates successful execution with no errors.
    
    Raises:
        FileNotFoundError: If Allrun script does not exist in case_dir
        RuntimeError: If Allrun script execution fails repeatedly
        TimeoutError: If execution exceeds timeout limit
    
    Example:
        >>> errors = run_allrun_and_collect_errors(
        ...     case_dir="/path/to/case",
        ...     timeout=1800,
        ...     max_retries=2
        ... )
        >>> if not errors:
        ...     print("Simulation completed successfully")
        >>> else:
        ...     print(f"Found {len(errors)} errors")
    """
    allrun_file_path = os.path.join(case_dir, "Allrun")
    if not os.path.exists(allrun_file_path):
        return [f"Allrun script not found at {allrun_file_path}"]

    allrun_script = _read_text(Path(allrun_file_path))
    preflight_errors = validate_openfoam_case_preflight(case_dir, allrun_script)
    if preflight_errors:
        return preflight_errors
    
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")

    # Cleanup artifacts from an earlier failed invocation before execution.
    _cleanup_run_artifacts(case_dir)

    last_error_logs = []

    # Run with retries
    for attempt in range(1, max_retries + 1):
        print(f"Running Allrun (attempt {attempt}/{max_retries})")
        command_result = run_command(
            allrun_file_path,
            out_file,
            err_file,
            case_dir,
            timeout,
        )

        # Inspect
        error_logs: List[Any] = []
        timed_out = bool(_result_field(command_result, "timed_out", False))
        returncode = _result_field(command_result, "returncode", 0)
        if timed_out:
            error_logs.append(_validation_error(
                "Allrun",
                f"Allrun exceeded the {timeout} second execution timeout.",
            ))
        elif returncode not in (None, 0):
            error_logs.append(_validation_error(
                "Allrun",
                f"Allrun exited with non-zero return code {returncode}.",
            ))

        error_logs.extend(check_foam_errors(case_dir))
        error_logs.extend(
            validate_openfoam_case_postflight(case_dir, allrun_script)
        )
        error_logs = _deduplicate_errors(error_logs)
        if len(error_logs) == 0:
            return []

        last_error_logs = error_logs
        if attempt < max_retries:
            print("Allrun reported errors; retrying after cleanup...")
            _cleanup_run_artifacts(case_dir)

    return last_error_logs


def run_simulation_local(
    case_id: str,
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1
) -> RunOut:
    """
    Run OpenFOAM simulation locally and return execution status.
    
    This function executes the Allrun script in the specified case directory
    and returns the execution status along with any job information.
    For local execution, job_id is always None.
    
    Args:
        case_id (str): Unique identifier for the case
        case_dir (str): Directory path containing the OpenFOAM case
        timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
    
    Returns:
        RunOut: Contains:
            - job_id (None): Always None for local execution
            - status (str): Execution status ("completed" or "failed")
    
    Raises:
        FileNotFoundError: If case directory or Allrun script does not exist
        RuntimeError: If simulation execution fails
    
    Example:
        >>> result = run_simulation_local(
        ...     case_id="test_case",
        ...     case_dir="/path/to/case",
        ...     timeout=1800
        ... )
        >>> print(f"Simulation status: {result.status}")
    """
    errors = run_allrun_and_collect_errors(case_dir, timeout, max_retries)
    status = "completed" if len(errors) == 0 else "failed"
    return RunOut(job_id=None, status=status)
