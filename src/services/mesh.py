import os
import re
import shutil
import subprocess
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from utils import save_file
from . import global_llm_service


def copy_custom_mesh(custom_mesh_path: str, user_requirement: str, case_dir: str) -> Dict[str, Any]:
    """
    Copy and process a custom mesh file for OpenFOAM simulation.
    
    This function copies a custom mesh file (typically .msh format) to the case directory,
    creates necessary OpenFOAM directories, generates a basic controlDict, and converts
    the mesh to OpenFOAM format using gmshToFoam.
    
    Args:
        custom_mesh_path (str): Path to the custom mesh file (.msh format)
        user_requirement (str): User requirements for generating controlDict
        case_dir (str): Directory path where the case files will be created
    
    Returns:
        Dict[str, Any]: Contains:
            - mesh_info (Dict[str, Any]): Mesh information including:
                - mesh_file_path (str): Path to the copied mesh file
                - mesh_file_type (str): Type of mesh file ("gmsh")
                - mesh_description (str): Description of the mesh
                - requires_blockmesh_removal (bool): Whether blockMesh should be removed
            - mesh_commands (List[str]): List of mesh validation commands
            - custom_mesh_used (bool): Whether custom mesh was used
            - error_logs (List[str]): List of any error messages
    
    Raises:
        FileNotFoundError: If custom mesh file does not exist
        RuntimeError: If gmshToFoam conversion fails
        ValueError: If mesh file is invalid
    
    Example:
        >>> result = copy_custom_mesh(
        ...     custom_mesh_path="/path/to/mesh.msh",
        ...     user_requirement="Simple flow simulation",
        ...     case_dir="/path/to/case"
        ... )
        >>> print(f"Mesh processed: {result['mesh_info']['mesh_file_path']}")
    """
    error_logs: List[str] = []
    if not custom_mesh_path:
        return {"mesh_info": None, "mesh_commands": [], "error_logs": ["No custom mesh path provided"]}
    if not os.path.exists(custom_mesh_path):
        return {"mesh_info": None, "mesh_commands": [], "error_logs": [f"Custom mesh not found: {custom_mesh_path}"]}

    mesh_in_case_dir = os.path.join(case_dir, "geometry.msh")
    shutil.copy2(custom_mesh_path, mesh_in_case_dir)

    constant_dir = os.path.join(case_dir, "constant")
    system_dir = os.path.join(case_dir, "system")
    os.makedirs(constant_dir, exist_ok=True)
    os.makedirs(system_dir, exist_ok=True)

    controldict_prompt = (
        f"<user_requirements>{user_requirement}</user_requirements>\n"
        "Please create a basic controlDict file for mesh conversion. "
        "The file should include only the essential settings needed for gmshToFoam to work. "
        "IMPORTANT: Return ONLY the complete controlDict file content without any additional text."
    )
    # Use global llm instance
    controldict_content = global_llm_service.invoke(controldict_prompt, (
        "You are an expert in OpenFOAM simulation setup. "
        "Create a minimal controlDict for gmshToFoam."
    )).strip()
    if controldict_content:
        save_file(os.path.join(system_dir, "controlDict"), controldict_content)

    # Convert mesh
    try:
        result = subprocess.run(["gmshToFoam", "geometry.msh"], cwd=case_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        return {"mesh_info": None, "mesh_commands": [], "error_logs": [f"gmshToFoam failed: {e.stderr}"]}

    polyMesh_dir = os.path.join(constant_dir, "polyMesh")
    if not os.path.exists(polyMesh_dir):
        return {"mesh_info": None, "mesh_commands": [], "error_logs": ["polyMesh directory not created"]}

    foam_file = os.path.join(case_dir, f"{os.path.basename(case_dir)}.foam")
    with open(foam_file, 'w') as f:
        pass

    return {
        "mesh_info": {
            "mesh_file_path": mesh_in_case_dir,
            "mesh_file_type": "gmsh",
            "mesh_description": "Custom mesh processed by preprocessor",
            "requires_blockmesh_removal": True,
        },
        "mesh_commands": ["checkMesh"],
        "custom_mesh_used": True,
        "error_logs": error_logs,
    }


def prepare_standard_mesh(user_requirement: str, case_dir: str) -> Dict[str, Any]:
    """
    Prepare standard mesh configuration for OpenFOAM simulation.
    
    This function returns a standard mesh configuration that indicates
    no custom mesh processing is required. It's used when the simulation
    will use standard OpenFOAM mesh generation tools like blockMesh.
    
    Args:
        user_requirement (str): User requirements (used for consistency)
        case_dir (str): Directory path for the case (used for consistency)
    
    Returns:
        Dict[str, Any]: Contains:
            - mesh_info (None): No custom mesh information
            - mesh_commands (List[str]): Empty list of mesh commands
            - mesh_file_destination (None): No mesh file destination
            - custom_mesh_used (bool): False, indicating standard mesh
            - error_logs (List[str]): Empty list of error messages
    
    Example:
        >>> result = prepare_standard_mesh(
        ...     user_requirement="Simple flow simulation",
        ...     case_dir="/path/to/case"
        ... )
        >>> print(f"Using standard mesh: {not result['custom_mesh_used']}")
    """
    return {
        "mesh_info": None,
        "mesh_commands": [],
        "mesh_file_destination": None,
        "custom_mesh_used": False,
        "error_logs": [],
    }



# ====================== GMSH mesh generation ======================

# Prompts used by LLM interactions for mesh-related steps
BOUNDARY_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM mesh processing and simulations. "
    "Your role is to analyze and modify boundary conditions in OpenFOAM polyMesh boundary file. "
    "You understand both 2D and 3D simulations and know how to properly set boundary conditions. "
    "For 2D simulations, you know which boundaries should be set to 'empty' type and 'empty' physicalType. "
    "You are precise and only return the exact boundary file content without any additional text or explanations. "
    "IMPORTANT: Only change the specified boundary to 'empty' type and leave all other boundaries exactly as they are."
)

CONTROLDICT_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation setup. "
    "Your role is to create a basic controlDict file for mesh conversion. "
    "You understand the minimal requirements needed for gmshToFoam to work. "
    "You are precise and only return the exact controlDict file content without any additional text or explanations."
)

GMSH_PYTHON_SYSTEM_PROMPT = (
    "You are an expert in GMSH Python API and OpenFOAM mesh generation. "
    "Your role is to create Python code that uses the GMSH library to generate meshes based on user requirements. "
    "You understand: "
    "- GMSH Python API for geometry creation "
    "- How to create points, lines, surfaces, and volumes programmatically "
    "- How to assign physical groups for OpenFOAM compatibility "
    "- How to control mesh sizing and refinement "
    "- How to handle 2D and 3D geometries "
    "You can: "
    "- Create complex geometries using GMSH Python API "
    "- Set up proper boundary conditions for OpenFOAM "
    "- Implement mesh refinement strategies "
    "- Generate 3D meshes with correct boundary assignments "
    "CRITICAL REQUIREMENTS: "
    "- Always generate 3D meshes for OpenFOAM simulations "
    "- Set mesh sizes and generate 3D mesh: gmsh.model.mesh.generate(3) "
    "- AFTER 3D mesh generation, identify surfaces using gmsh.model.getEntities(2) "
    "- Use gmsh.model.getBoundingBox(dim, tag) to analyze surface positions and categorize them "
    "- Do not use gmsh.model.getCenterOfMass(dim, tag) function to analyze surface positions "
    "- Create 2D physical groups based on spatial analysis, not during geometry creation "
    "- Use user-specified boundary names "
    "- Create physical groups for all surfaces and the volume domain "
    "- Set gmsh.option.setNumber('Mesh.MshFileVersion', 2.2) for OpenFOAM compatibility "
    "- Save as 'geometry.msh' and finalize GMSH "
    "- Use proper coordinate system - define z_min and z_max variables and use them consistently for boundary detection "
    "- Use bounding box coordinates (x_min, y_min, z_min, x_max, y_max, z_max) directly for boundary detection, NOT center points "
    "- Ensure all boundary types (example: inlet, outlet, top, bottom, cylinder, frontAndBack) are properly detected and created "
    "CRITICAL ORDER: Create geometry then Extrude then Synchronize then Generate mesh then create physical groups "
    "CRITICAL: Use bounding box coordinates consistently for ALL boundaries - do not mix center points and bounding box coordinates in the same boundary detection logic "
    "MOST CRITICAL: NEVER create physical groups before mesh generation. Always create them AFTER gmsh.model.mesh.generate(3) "
    "MOST CRITICAL: Physical groups created before mesh generation will reference wrong surface tags after extrusion and meshing "
    "CRITICAL FACE DETECTION: "
    "- For thin boundary surfaces: use abs(zmin - zmax) < tol AND (abs(zmin - z_min) < tol OR abs(zmin - z_max) < tol) "
    "- Thin surfaces at z_min and z_max are boundary surfaces that need physical groups "
    "- Use tolerance tol = 1e-6 for floating point comparisons "
    "- Ensure ALL user-specified boundaries are detected and assigned to physical groups "
    "IMPORTANT: Use your expertise to create robust, adaptable code that can handle various geometry types and boundary conditions."
)

BOUNDARY_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM mesh generation and boundary condition analysis. "
    "Your role is to extract boundary names from user requirements for mesh generation. "
    "You understand: "
    "- Common OpenFOAM boundary types (inlet, outlet, wall, cylinder, etc.) "
    "- How to identify boundary names from natural language descriptions "
    "- The importance of accurate boundary identification for mesh generation "
    "You can: "
    "- Parse user requirements to identify all mentioned boundaries "
    "- Distinguish between boundary names and other geometric terms "
    "- Handle variations in boundary naming conventions "
    "- Return a clean list of boundary names "
    "IMPORTANT: Return ONLY a comma-separated list of boundary names without any additional text, explanations, or formatting. "
    "Example: inlet,outlet,wall,cylinder "
    "If no boundaries are mentioned, return an empty string."
)

GMSH_PYTHON_ERROR_CORRECTION_SYSTEM_PROMPT = (
    "You are an expert in debugging GMSH Python API code. "
    "Your role is to analyze GMSH Python errors and fix the corresponding code. "
    "You understand common GMSH Python API errors including: "
    "- Geometry definition errors (invalid points, lines, surfaces, volumes) "
    "- Physical group assignment issues "
    "- Mesh generation problems "
    "- API usage errors "
    "- Missing boundary definitions that cause OpenFOAM conversion failures "
    "- Mesh quality issues detected by checkMesh (skewness, aspect ratio, etc.) "
    "You can identify the root cause of errors and provide corrected Python code. "
    "CRITICAL REQUIREMENTS: "
    "- Ensure 3D mesh generation for OpenFOAM compatibility "
    "- Use proper spatial analysis for boundary identification "
    "- Create complete physical group definitions for surfaces and volumes "
    "- Handle various geometry types and boundary conditions "
    "- When missing boundaries are mentioned, ensure they are properly defined "
    "- Do not use gmsh.model.getCenterOfMass(dim, tag) function to analyze surface positions "
    "- Address mesh quality issues by adjusting mesh sizing and refinement strategies "
    "CRITICAL CORRECTIONS: "
    "- Use proper coordinate system variables (z_min, z_max) for boundary detection "
    "- Use bounding box coordinates directly for boundary detection, NOT center points "
    "- Ensure all boundary types are detected: (example: inlet, outlet, top, bottom, cylinder, frontAndBack) "
    "- Check boundary detection logic for coordinate system consistency "
    "- Verify that extrusion creates proper 3D geometry with all expected surfaces "
    "- For mesh quality issues: adjust mesh sizes, add refinement zones, improve geometry definition "
    "CRITICAL ORDER: Create geometry then Extrude then Synchronize then Generate mesh then create physical groups "
    "CRITICAL: Use bounding box coordinates consistently for ALL boundary types - do not mix center points and bounding box coordinates in the same boundary detection logic "
    "MOST CRITICAL FIX: If boundaries are missing after gmshToFoam, move ALL physical group creation to AFTER gmsh.model.mesh.generate(3) "
    "MOST CRITICAL FIX: Physical groups created before mesh generation will have wrong surface tag references "
    "CRITICAL FACE DETECTION FIXES: "
    "- Fix thin boundary detection: use abs(zmin - zmax) < tol AND (abs(zmin - z_min) < tol OR abs(zmin - z_max) < tol) "
    "- Use tolerance tol = 1e-6 for all floating point comparisons "
    "- Ensure thin surfaces at z_min and z_max are properly classified as boundary surfaces "
    "- Check that ALL user-specified boundaries are detected and assigned to physical groups "
    "MESH QUALITY FIXES: "
    "- For high skewness: refine mesh in problematic areas, adjust element sizes "
    "- For poor aspect ratio: use smaller mesh sizes, add refinement zones "
    "- For non-orthogonality: improve geometry definition, use structured meshing where possible "
    "- For negative volume elements: check geometry validity, ensure proper surface orientation "
    "IMPORTANT: Use your expertise to diagnose and fix issues while maintaining code adaptability for different problems."
)


class GMSHPythonCode(BaseModel):
    python_code: str = Field(description="Complete Python code using GMSH library")
    mesh_type: str = Field(description="Type of mesh (2D or 3D)")
    geometry_type: str = Field(description="Type of geometry being created")


class GMSHPythonCorrection(BaseModel):
    corrected_code: str = Field(description="Corrected GMSH Python code")
    error_analysis: str = Field(description="Analysis of the error and what was fixed")


def extract_boundary_names_from_requirements(user_requirement: str) -> List[str]:
    try:
        extraction_prompt = (
            f"<user_requirements>{user_requirement}</user_requirements>\n"
            "Please extract all boundary names mentioned in the user requirements. "
            "Look for terms like inlet, outlet, wall, cylinder, top, bottom, front, back, side, etc. "
            "Focus on boundaries that would need to be defined in the mesh for OpenFOAM simulation. "
            "Return ONLY a comma-separated list of boundary names without any additional text."
        )
        boundary_response = global_llm_service.invoke(extraction_prompt, BOUNDARY_EXTRACTION_SYSTEM_PROMPT).strip()
        if boundary_response:
            return [name.strip() for name in boundary_response.split(',') if name.strip()]
        return []
    except Exception:
        # Fallback keyword search
        requirement_lower = (user_requirement or "").lower()
        boundary_keywords = ['inlet', 'outlet', 'wall', 'cylinder', 'top', 'bottom', 'front', 'back', 'side']
        return [k for k in boundary_keywords if k in requirement_lower]


def check_boundary_file_for_missing_boundaries(boundary_file_path: str, expected_boundaries: List[str]):
    if not os.path.exists(boundary_file_path):
        return False, expected_boundaries, []
    try:
        with open(boundary_file_path, 'r') as f:
            content = f.read()
        boundary_pattern = r'(\w+)\s*\{'
        found_boundaries = re.findall(boundary_pattern, content)
        boundary_keywords = ['type', 'physicalType', 'nFaces', 'startFace', 'FoamFile']
        found_boundaries = [b for b in found_boundaries if b not in boundary_keywords]
        missing_boundaries = [b for b in expected_boundaries if b not in found_boundaries]
        return len(missing_boundaries) == 0, missing_boundaries, found_boundaries
    except Exception:
        return False, expected_boundaries, []


def _correct_gmsh_python_code(user_requirement: str, current_code: str, error_output: str, found_boundaries=None, expected_boundaries=None):
    try:
        is_boundary_mismatch = isinstance(error_output, str) and "Boundary mismatch after gmshToFoam" in error_output
        boundary_info = ""
        if is_boundary_mismatch and found_boundaries is not None and expected_boundaries is not None:
            boundary_info = (
                f"\n<boundary_mismatch>Found boundaries in OpenFOAM: {found_boundaries}. "
                f"Expected boundaries: {expected_boundaries}. "
                "Please correct the mesh code so that the boundaries in the OpenFOAM boundary file match the expected boundaries exactly."
                "Note that these boundaries might be present in the msh file, but not in the boundary file after running gmshToFoam to convert the msh file to OpenFOAM format."
                "MOST LIKELY CAUSES: "
                "1. Physical groups were created before mesh generation. Move ALL physical group creation to AFTER gmsh.model.mesh.generate(3). "
                "2. Points created at z=0 instead of z=z_min. Create ALL points at z=z_min for proper boundary detection. "
                "3. Incorrect surface detection logic - surfaces not properly classified by position. "
                "4. If 'defaultFaces' appears, it means some surfaces weren't assigned to any physical group. "
                "5. Check that ALL surfaces are being classified and assigned to the correct user-specified boundary names."
                "</boundary_mismatch>"
            )
        if is_boundary_mismatch:
            correction_prompt = (
                f"<user_requirements>{user_requirement}</user_requirements>{boundary_info}\n"
                f"<current_python_code>{current_code}</current_python_code>\n"
                "Please analyze the current Python code and the boundary mismatch information. "
                "The mesh generation was successful, but the boundaries in the OpenFOAM conversion do not match the expected boundaries. "
                "MOST LIKELY SOLUTIONS: "
                "1. Move ALL physical group creation to AFTER gmsh.model.mesh.generate(3). "
                "2. Use correct thin boundary detection: abs(zmin - zmax) < tol AND (abs(zmin - z_min) < tol OR abs(zmin - z_max) < tol). "
                "3. Use tolerance tol = 1e-6 for all floating point comparisons. "
                "4. Use exact boundary names from user requirements, do not hardcode specific names. "
                "Provide a corrected Python code that ensures the boundaries in the OpenFOAM boundary file match the expected boundaries exactly. "
                "IMPORTANT: Return ONLY the complete corrected Python code without any additional text."
            )
        else:
            correction_prompt = (
                f"<user_requirements>{user_requirement}</user_requirements>\n"
                f"<current_python_code>{current_code}</current_python_code>\n"
                f"<gmsh_python_error_output>{error_output}</gmsh_python_error_output>\n"
                "Please analyze the GMSH Python error output and the current Python code. "
                "Identify the specific error and provide a corrected Python code that fixes the issue. "
                "IMPORTANT: Return ONLY the complete corrected Python code without any additional text."
            )
        correction_response = global_llm_service.invoke(
            correction_prompt,
            GMSH_PYTHON_ERROR_CORRECTION_SYSTEM_PROMPT,
            pydantic_obj=GMSHPythonCorrection,
        )
        if correction_response.corrected_code:
            return correction_response.corrected_code
    except Exception:
        pass
    return None


def run_checkmesh_and_correct(case_dir: str, python_file: str, max_loop: int, current_loop: int) -> Tuple[bool, bool, str]:
    """Run checkMesh and optionally generate corrected code. Returns (success, should_continue, corrected_code)."""
    try:
        result = subprocess.run(["checkMesh"], cwd=case_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        checkmesh_output = result.stdout
        if "Failed" in checkmesh_output and "mesh checks" in checkmesh_output:
            failed_match = re.search(r"Failed (\d+) mesh checks", checkmesh_output)
            if failed_match and current_loop < max_loop:
                with open(python_file, 'r') as f:
                    current_code = f.read()
                checkmesh_error = (
                    f"checkMesh output:\n{checkmesh_output}\n"
                    "Please analyze the checkMesh output and correct the mesh generation code. "
                    "Common issues include poor mesh quality, geometry issues, boundary layer problems, and boundary naming mismatch."
                )
                corrected_code = _correct_gmsh_python_code("", current_code, checkmesh_error)
                if corrected_code:
                    return False, True, corrected_code
            return False, False, ""
        return True, False, ""
    except subprocess.CalledProcessError as e:
        if current_loop < max_loop:
            return False, True, ""
        return False, False, ""
    except Exception:
        return False, False, ""


def handle_gmsh_mesh(
    user_requirement: str,
    case_dir: str,
    max_loop: int = 3
) -> Dict[str, Any]:
    """
    Generate GMSH mesh for OpenFOAM simulation using Python API.
    
    This function creates a Python script that uses the GMSH library to generate
    a 3D mesh based on user requirements. It handles geometry creation, mesh
    generation, boundary detection, and OpenFOAM conversion with error correction.
    
    Args:
        user_requirement (str): Natural language description of the simulation geometry
        case_dir (str): Directory path where mesh files will be created
        max_loop (int, optional): Maximum number of retry attempts for error correction. Defaults to 3.
    
    Returns:
        Dict[str, Any]: Contains:
            - mesh_info (Dict[str, Any]): Mesh information including:
                - mesh_file_path (str): Path to the generated .msh file
                - mesh_file_type (str): Type of mesh file ("gmsh")
                - mesh_description (str): Description of the generated mesh
                - requires_blockmesh_removal (bool): Whether blockMesh should be removed
            - mesh_commands (List[str]): List of mesh validation commands
            - mesh_file_destination (str): Path to the mesh file
            - custom_mesh_used (bool): Whether custom mesh was used
            - error_logs (List[str]): List of any error messages
    
    Raises:
        ValueError: If user requirements cannot be parsed
        RuntimeError: If GMSH Python generation fails after max_loop attempts
        FileNotFoundError: If required directories cannot be created
    
    Example:
        >>> result = handle_gmsh_mesh(
        ...     user_requirement="Create a 3D channel flow with inlet and outlet",
        ...     case_dir="/path/to/case",
        ...     max_loop=3
        ... )
        >>> print(f"Mesh generated: {result['mesh_info']['mesh_file_path']}")
    """
    case_dir = os.path.abspath(case_dir)
    error_logs: List[str] = []
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)

    python_file = os.path.join(case_dir, "generate_mesh.py")
    msh_file = os.path.join(case_dir, "geometry.msh")

    expected_boundaries = extract_boundary_names_from_requirements(user_requirement)

    gmsh_python_current_loop = 0
    corrected_python_code = None

    while gmsh_python_current_loop < max_loop:
        gmsh_python_current_loop += 1
        should_generate_new_code = corrected_python_code is None
        try:
            if should_generate_new_code:
                missing_boundary_info = ""
                python_prompt = (
                    f"<user_requirements>{user_requirement}</user_requirements>\n"
                    f"{missing_boundary_info}"
                    "Please create Python code using the GMSH library to generate a mesh based on the user requirements. "
                    "Use boundary names specified in user requirements (e.g., 'inlet', 'outlet', 'wall', 'cylinder', etc.). "
                    "Return ONLY the complete Python code without any additional text."
                )
                python_response = global_llm_service.invoke(python_prompt, GMSH_PYTHON_SYSTEM_PROMPT, pydantic_obj=GMSHPythonCode)  # type: ignore
                if not python_response.python_code:
                    if gmsh_python_current_loop >= max_loop:
                        return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None, "error_logs": error_logs}
                    continue
                python_code_to_use = python_response.python_code
                geometry_type = python_response.geometry_type
            else:
                python_code_to_use = corrected_python_code
                geometry_type = "corrected"

            save_file(python_file, python_code_to_use)
            corrected_python_code = None

            process = subprocess.Popen(["python", python_file], cwd=case_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
            return_code = process.wait()
            stderr_output = process.stderr.read()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args, stderr=stderr_output)

            if not os.path.exists(msh_file):
                if stderr_output and gmsh_python_current_loop < max_loop:
                    corrected = _correct_gmsh_python_code(user_requirement, python_code_to_use, stderr_output)
                    if corrected:
                        corrected_python_code = corrected
                        continue
                if gmsh_python_current_loop >= max_loop:
                    return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None, "error_logs": error_logs}
                continue

            # Preprocess for OpenFOAM conversion
            constant_dir = os.path.join(case_dir, "constant")
            system_dir = os.path.join(case_dir, "system")
            os.makedirs(constant_dir, exist_ok=True)
            os.makedirs(system_dir, exist_ok=True)
            controldict_prompt = (
                f"<user_requirements>{user_requirement}</user_requirements>\n"
                "Please create a basic controlDict file for mesh conversion. "
                "The file should include only the essential settings needed for gmshToFoam to work. "
                "IMPORTANT: Return ONLY the complete controlDict file content without any additional text."
            )
            controldict_content = global_llm_service.invoke(controldict_prompt, CONTROLDICT_SYSTEM_PROMPT).strip()  # type: ignore
            if controldict_content:
                save_file(os.path.join(system_dir, "controlDict"), controldict_content)

            result = subprocess.run(["gmshToFoam", "geometry.msh"], cwd=case_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            polyMesh_dir = os.path.join(constant_dir, "polyMesh")
            if not os.path.exists(polyMesh_dir):
                raise subprocess.CalledProcessError(1, "gmshToFoam", "polyMesh directory not created")

            boundary_file = os.path.join(polyMesh_dir, "boundary")
            if os.path.exists(boundary_file):
                all_present, missing_boundaries, found_boundaries = check_boundary_file_for_missing_boundaries(boundary_file, expected_boundaries)
                if set(found_boundaries) != set(expected_boundaries):
                    if gmsh_python_current_loop < max_loop:
                        with open(python_file, 'r') as f:
                            current_code = f.read()
                        boundary_error = (
                            f"Boundary mismatch after gmshToFoam. Found boundaries: {found_boundaries}. Expected boundaries: {expected_boundaries}. "
                        )
                        corrected = _correct_gmsh_python_code(user_requirement, current_code, boundary_error, found_boundaries, expected_boundaries)
                        if corrected:
                            corrected_python_code = corrected
                            continue
                    else:
                        return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None, "error_logs": error_logs}

                # Mesh quality check and possible correction
                ok, should_continue, corrected = run_checkmesh_and_correct(case_dir, python_file, max_loop, gmsh_python_current_loop)  # type: ignore
                if not ok:
                    if should_continue and corrected:
                        corrected_python_code = corrected
                        continue
                    if should_continue:
                        continue
                    return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None, "error_logs": error_logs}

                # Boundary update as per requirements
                with open(boundary_file, 'r') as f:
                    boundary_content = f.read()
                boundary_prompt = (
                    f"<user_requirements>{user_requirement}</user_requirements>\n"
                    f"<boundary_file_content>{boundary_content}</boundary_file_content>\n"
                    "Please analyze the user requirements and boundary file content. "
                    "Identify which boundary is to be modified based on the boundaries mentioned in the user requirements."
                    "If this is a 2D simulation, modify ONLY the appropriate boundary to 'empty' type and 'empty' physicalType. "
                    "Based on the no slip boundaries mentioned in the user requirements, modify the appropriate boundary/boundaries to type 'wall' and physicalType 'wall'. "
                    "If this is a 3D simulation, only modify the appropriate boundary/boundaries to type 'wall' and physicalType 'wall'."
                    "IMPORTANT: Do not change any other boundaries - leave them exactly as they are. "
                    "Return ONLY the complete boundary file content with any necessary modifications. No additional text."
                )
                updated_boundary_content = global_llm_service.invoke(boundary_prompt, BOUNDARY_SYSTEM_PROMPT).strip()  # type: ignore
                if updated_boundary_content:
                    save_file(boundary_file, updated_boundary_content)

            # Create .foam file and return info
            foam_file = os.path.join(case_dir, f"{os.path.basename(case_dir)}.foam")
            with open(foam_file, 'w'):
                pass

            mesh_commands: List[str] = []
            return {
                "mesh_info": {
                    "mesh_file_path": msh_file,
                    "mesh_file_type": "gmsh",
                    "mesh_description": f"GMSH generated {geometry_type} mesh",
                    "requires_blockmesh_removal": True,
                },
                "mesh_commands": mesh_commands,
                "mesh_file_destination": msh_file,
                "custom_mesh_used": True,
                "error_logs": error_logs,
            }
        except subprocess.CalledProcessError as e:
            if gmsh_python_current_loop < max_loop:
                try:
                    with open(python_file, 'r') as f:
                        current_code = f.read()
                    corrected = _correct_gmsh_python_code(user_requirement, current_code, e.stderr)
                    if corrected:
                        corrected_python_code = corrected
                        continue
                except Exception:
                    pass
            if gmsh_python_current_loop >= max_loop:
                return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None, "error_logs": error_logs}
        except Exception:
            if gmsh_python_current_loop >= max_loop:
                return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None, "error_logs": error_logs}
            continue

    return {"mesh_info": None, "mesh_commands": [], "mesh_file_destination": None, "error_logs": error_logs}

