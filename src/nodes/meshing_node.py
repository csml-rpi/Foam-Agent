import os
import shutil
import subprocess
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import save_file, remove_file

# System prompts for LLM interactions
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

class MeshInfoPydantic(BaseModel):
    mesh_file_path: str = Field(description="Path to the custom mesh file (e.g., .msh, .stl, .obj)")
    mesh_file_type: str = Field(description="Type of mesh file (gmsh, stl, obj, etc.)")
    mesh_description: str = Field(description="Description of the mesh and any specific requirements")
    requires_blockmesh_removal: bool = Field(description="Whether to remove blockMeshDict file", default=True)

class MeshCommandsPydantic(BaseModel):
    mesh_commands: List[str] = Field(description="List of OpenFOAM commands needed to process the custom mesh")
    mesh_file_destination: str = Field(description="Destination path for the mesh file in the case directory")

class GMSHPythonCode(BaseModel):
    python_code: str = Field(description="Complete Python code using GMSH library")
    mesh_type: str = Field(description="Type of mesh (2D or 3D)")
    geometry_type: str = Field(description="Type of geometry being created")

class GMSHPythonCorrection(BaseModel):
    corrected_code: str = Field(description="Corrected GMSH Python code")
    error_analysis: str = Field(description="Analysis of the error and what was fixed")

def _correct_gmsh_python_code(state, current_code, error_output):
    """
    Attempt to correct GMSH Python code based on error output or missing boundary info.
    
    Args:
        state: State object containing LLM service
        current_code: Current Python code content
        error_output: GMSH Python error output or missing boundary error message
    
    Returns:
        Corrected Python code content or None if correction failed
    """
    try:
        # Detect if this is a boundary mismatch error
        is_boundary_mismatch = (
            isinstance(error_output, str) and "Boundary mismatch after gmshToFoam" in error_output
        )
        found_boundaries = state['found_boundaries']
        expected_boundaries = state['expected_boundaries']
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
                f"<user_requirements>{state['user_requirement']}</user_requirements>{boundary_info}\n"
                f"<current_python_code>{current_code}</current_python_code>\n"
                "Please analyze the current Python code and the boundary mismatch information. "
                "The mesh generation was successful, but the boundaries in the OpenFOAM conversion do not match the expected boundaries. "
                "Note that these boundaries might be present in the msh file, but not in the boundary file after running gmshToFoam to convert the msh file to OpenFOAM format."
                "MOST LIKELY SOLUTIONS: "
                "1. Move ALL physical group creation to AFTER gmsh.model.mesh.generate(3). "
                "2. Use correct thin boundary detection: abs(zmin - zmax) < tol AND (abs(zmin - z_min) < tol OR abs(zmin - z_max) < tol). "
                "3. Use tolerance tol = 1e-6 for all floating point comparisons. "
                "4. Use exact boundary names from user requirements, do not hardcode specific names. "
                "Physical groups created before mesh generation will reference wrong surface tags after extrusion and meshing."
                "Provide a corrected Python code that ensures the boundaries in the OpenFOAM boundary file match the expected boundaries exactly. "
                "IMPORTANT: Return ONLY the complete corrected Python code without any additional text."
            )
        else:
            # Fallback to previous logic for other errors
            missing_boundary_info = ""
            if 'missing_boundaries' in state and state['missing_boundaries']:
                missing_boundary_info = (
                    f"\n<missing_boundaries>Previous attempts were missing these boundaries: {state['missing_boundaries']}. "
                    "Ensure these boundaries are properly defined in the GMSH physical groups.</missing_boundaries>"
                )
            correction_prompt = (
                f"<user_requirements>{state['user_requirement']}</user_requirements>{missing_boundary_info}\n"
                f"<current_python_code>{current_code}</current_python_code>\n"
                f"<gmsh_python_error_output>{error_output}</gmsh_python_error_output>\n"
                "Please analyze the GMSH Python error output and the current Python code. "
                "Identify the specific error and provide a corrected Python code that fixes the issue. "
                "IMPORTANT: Return ONLY the complete corrected Python code without any additional text."
            )
        correction_response = state["llm_service"].invoke(
            correction_prompt, 
            GMSH_PYTHON_ERROR_CORRECTION_SYSTEM_PROMPT, 
            pydantic_obj=GMSHPythonCorrection
        )
        if correction_response.corrected_code:
            print(f"Error analysis: {correction_response.error_analysis}")
            return correction_response.corrected_code
    except Exception as e:
        print(f"Error in Python code correction: {str(e)}")
    return None

def extract_boundary_names_from_requirements(state, user_requirement):
    """Extract boundary names mentioned in user requirements using LLM."""
    try:
        extraction_prompt = (
            f"<user_requirements>{user_requirement}</user_requirements>\n"
            "Please extract all boundary names mentioned in the user requirements. "
            "Look for terms like inlet, outlet, wall, cylinder, top, bottom, front, back, side, etc. "
            "Focus on boundaries that would need to be defined in the mesh for OpenFOAM simulation. "
            "Return ONLY a comma-separated list of boundary names without any additional text."
        )
        
        boundary_response = state["llm_service"].invoke(extraction_prompt, BOUNDARY_EXTRACTION_SYSTEM_PROMPT).strip()
        
        if boundary_response:
            boundary_names = [name.strip() for name in boundary_response.split(',') if name.strip()]
            print(f"LLM extracted boundary names: {boundary_names}")
            return boundary_names
        else:
            print("No boundary names extracted by LLM")
            return []
            
    except Exception as e:
        print(f"Error in LLM boundary extraction: {e}")
        # Fallback to simple keyword matching
        boundary_keywords = ['inlet', 'outlet', 'wall', 'cylinder', 'top', 'bottom', 'front', 'back', 'side']
        found_boundaries = []
        
        requirement_lower = user_requirement.lower()
        for keyword in boundary_keywords:
            if keyword in requirement_lower:
                found_boundaries.append(keyword)
        
        print(f"Fallback boundary extraction: {found_boundaries}")
        return found_boundaries

def check_boundary_file_for_missing_boundaries(boundary_file_path, expected_boundaries):
    """Check if all expected boundaries are present in the boundary file."""
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
        all_present = len(missing_boundaries) == 0
        
        return all_present, missing_boundaries, found_boundaries
        
    except Exception as e:
        print(f"Error reading boundary file: {e}")
        return False, expected_boundaries, []

def handle_custom_mesh(state, case_dir):
    print("============================== Custom Mesh Processing ==============================")
    custom_mesh_path = state.get("custom_mesh_path")
    error_logs = []
    if not custom_mesh_path:
        error_logs.append("No custom mesh path provided in state")
        print("Error: No custom mesh path provided")
        return {
            "mesh_info": None,
            "mesh_commands": [],
            "error_logs": error_logs
        }
    if not os.path.exists(custom_mesh_path):
        error_logs.append(f"Custom mesh file does not exist: {custom_mesh_path}")
        print(f"Error: Custom mesh file not found at {custom_mesh_path}")
        return {
            "mesh_info": None,
            "mesh_commands": [],
            "error_logs": error_logs
        }
    mesh_in_case_dir = os.path.join(case_dir, "geometry.msh")
    try:
        shutil.copy2(custom_mesh_path, mesh_in_case_dir)
        print(f"Copied custom mesh from {custom_mesh_path} to {mesh_in_case_dir}")
    except Exception as e:
        error_logs.append(f"Failed to copy custom mesh file: {str(e)}")
        print(f"Error: Failed to copy custom mesh file: {str(e)}")
        return {
            "mesh_info": None,
            "mesh_commands": [],
            "error_logs": error_logs
        }
    print(f"Using mesh file: {mesh_in_case_dir}")
    try:
        constant_dir = os.path.join(case_dir, "constant")
        os.makedirs(constant_dir, exist_ok=True)
        system_dir = os.path.join(case_dir, "system")
        os.makedirs(system_dir, exist_ok=True)
        controldict_prompt = (
            f"<user_requirements>{state['user_requirement']}</user_requirements>\n"
            "Please create a basic controlDict file for mesh conversion. "
            "The file should include only the essential settings needed for gmshToFoam to work. "
            "Use the application name as mentioned in the user requirements. "
            "IMPORTANT: Return ONLY the complete controlDict file content without any additional text."
        )
        controldict_content = state["llm_service"].invoke(controldict_prompt, CONTROLDICT_SYSTEM_PROMPT).strip()
        if controldict_content:
            controldict_file = os.path.join(system_dir, "controlDict")
            save_file(controldict_file, controldict_content)
            print("Created basic controlDict file for mesh conversion")
        result = subprocess.run(
            ["gmshToFoam", "geometry.msh"],
            cwd=case_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"gmshToFoam command output: {result.stdout}")
        polyMesh_dir = os.path.join(constant_dir, "polyMesh")
        if not os.path.exists(polyMesh_dir):
            error_logs.append("Mesh conversion failed: polyMesh directory not created")
            return {
                "mesh_info": None,
                "mesh_commands": [],
                "error_logs": error_logs
            }
        boundary_file = os.path.join(polyMesh_dir, "boundary")
        if os.path.exists(boundary_file):
            with open(boundary_file, 'r') as f:
                boundary_content = f.read()
            boundary_prompt = (
                f"<user_requirements>{state['user_requirement']}</user_requirements>\n"
                f"<boundary_file_content>{boundary_content}</boundary_file_content>\n"
                "Please analyze the user requirements and boundary file content. "
                "Identify which boundary is to be modified based on the boundaries mentioned in the user requirements."
                "If this is a 2D simulation, modify ONLY the appropriate boundary to 'empty' type and 'empty' physicalType. "
                "Based on the no slip boundaries mentioned in the user requirements, modify the appropriate boundary/boundaries to type 'wall' and physicalType 'wall'. "
                "If this is a 3D simulation, only modify the appropriate boundary/boundaries to type 'wall' and physicalType 'wall'."
                "IMPORTANT: Do not change any other boundaries - leave them exactly as they are. "
                "Return ONLY the complete boundary file content with any necessary modifications. No additional text."
            )
            updated_boundary_content = state["llm_service"].invoke(boundary_prompt, BOUNDARY_SYSTEM_PROMPT).strip()
            if updated_boundary_content:
                save_file(boundary_file, updated_boundary_content)
                print("Boundary file updated based on simulation requirements")
        foam_file = os.path.join(case_dir, f"{os.path.basename(case_dir)}.foam")
        with open(foam_file, 'w') as f:
            pass
        mesh_commands = [
            "checkMesh",
        ]
        return {
            "mesh_info": {
                "mesh_file_path": mesh_in_case_dir,
                "mesh_file_type": "gmsh",
                "mesh_description": "Custom mesh processed by preprocessor",
                "requires_blockmesh_removal": True
            },
            "mesh_commands": mesh_commands,
            "custom_mesh_used": True,
            "error_logs": error_logs
        }
    except subprocess.CalledProcessError as e:
        error_message = f"Error in mesh conversion: {str(e)}"
        if e.stdout:
            error_message += f"\nSTDOUT:\n{e.stdout}"
        if e.stderr:
            error_message += f"\nSTDERR:\n{e.stderr}"
        error_logs.append(error_message)
        return {
            "mesh_info": None,
            "mesh_commands": [],
            "error_logs": error_logs
        }

def run_checkmesh_and_correct(state, case_dir, python_file, max_loop, current_loop, corrected_code, error_logs):
    """
    Run checkMesh command and handle mesh quality issues.
    
    Args:
        state: State object containing LLM service and configuration
        case_dir: Case directory path
        python_file: Path to the Python mesh generation file
        max_loop: Maximum number of retry attempts
    
    Returns:
        tuple: (success: bool, should_continue: bool)
    """
    print("Running checkMesh to verify mesh quality...")
    
    try:
        # Run checkMesh command
        result = subprocess.run(
            ["checkMesh"],
            cwd=case_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        checkmesh_output = result.stdout
        print("checkMesh completed successfully")
        print(f"checkMesh output:\n{checkmesh_output}")
        
        # Check if checkMesh reported any failures
        if "Failed" in checkmesh_output and "mesh checks" in checkmesh_output:
            # Extract the number of failed checks
            failed_match = re.search(r"Failed (\d+) mesh checks", checkmesh_output)
            if failed_match:
                failed_count = int(failed_match.group(1))
                print(f"checkMesh detected {failed_count} mesh quality issues")
                
                if current_loop < max_loop:
                    print("Attempting to correct mesh generation based on checkMesh results...")
                    
                    # Read current Python code
                    with open(python_file, 'r') as f:
                        current_code = f.read()
                    
                    # Create error message for correction
                    checkmesh_error = (
                        f"checkMesh output:\n{checkmesh_output}\n"
                        "Please analyze the checkMesh output and correct the mesh generation code. "
                        "Common issues include: "
                        "- Poor mesh quality (skewness, aspect ratio, etc.) "
                        "- Geometry issues affecting mesh generation "
                        "- Boundary layer problems "
                        "- Boundary naming overlap or mismatch "
                        "Provide corrected Python code that addresses the specific mesh quality issues identified by checkMesh."
                    )
                    
                    # Use the existing correction function
                    corrected_code = _correct_gmsh_python_code(
                        state,
                        current_code,
                        checkmesh_error
                    )
                    
                    if corrected_code:
                        state['corrected_python_code'] = corrected_code
                        print("Generated corrected Python code for next attempt (checkMesh issues)")
                        return False, True  # Not successful, but should continue
                    else:
                        print("Failed to generate corrected code for checkMesh issues")
                        return False, False  # Not successful, should not continue
                else:
                    print(f"Failed to resolve checkMesh issues after {max_loop} attempts")
                    return False, False  # Not successful, should not continue
            else:
                print("checkMesh output contains 'Failed' but couldn't parse failure count")
                return False, False
        else:
            print("checkMesh passed - no mesh quality issues detected")
            return True, False  # Successful, no need to continue
            
    except subprocess.CalledProcessError as e:
        error_message = f"Error running checkMesh: {str(e)}"
        if e.stdout:
            error_message += f"\nSTDOUT:\n{e.stdout}"
        if e.stderr:
            error_message += f"\nSTDERR:\n{e.stderr}"
        print(error_message)
        state["error_logs"].append(error_message)
        
        if current_loop < max_loop:
            print("Retrying mesh generation due to checkMesh error...")
            return False, True  # Not successful, but should continue
        else:
            print(f"Failed to run checkMesh after {max_loop} attempts")
            return False, False  # Not successful, should not continue
    
    except Exception as e:
        print(f"Unexpected error in checkMesh: {str(e)}")
        state["error_logs"].append(f"Unexpected error in checkMesh: {str(e)}")
        return False, False

def handle_gmsh_mesh(state, case_dir):
    """Handle GMSH mesh generation using gmsh python logic."""
    print("============================== GMSH Mesh Generation ==============================")
    
    # Ensure case_dir exists
    case_dir = os.path.abspath(case_dir)
    error_logs = []
    if os.path.exists(case_dir):
        print(f"Warning: Case directory {case_dir} already exists. Overwriting.")
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    print(f"Created case directory: {case_dir}")
    
    # Define file paths
    python_file = os.path.join(case_dir, "generate_mesh.py")
    msh_file = os.path.join(case_dir, "geometry.msh")
    
    # Extract expected boundary names from user requirements
    expected_boundaries = extract_boundary_names_from_requirements(state, state["user_requirement"])
    print(f"Expected boundaries from user requirements: {expected_boundaries}")
    
    # Initialize loop counter if not present
    gmsh_python_current_loop = 0
    
    # Initialize missing boundaries tracking
    missing_boundaries = []
    
    # Initialize corrected code flag
    corrected_python_code = None
    
    max_loop = state['config'].max_loop
    while gmsh_python_current_loop < max_loop:
        gmsh_python_current_loop += 1
        print(f"GMSH Python attempt {gmsh_python_current_loop} of {max_loop}")
        
        # Determine if we should generate new code or use corrected code
        should_generate_new_code = True
        if corrected_python_code:
            should_generate_new_code = False
            python_code_to_use = corrected_python_code
            print("Using corrected Python code from previous attempt")
        
        try:
            if should_generate_new_code:
                # Generate Python code based on user requirements
                missing_boundary_info = ""
                if missing_boundaries:
                    missing_boundary_info = f"\n<missing_boundaries>Previous attempts were missing these boundaries: {missing_boundaries} in boundary file after performing gmshToFoam. Ensure these boundaries are properly defined in the GMSH physical groups.</missing_boundaries>"
                
                python_prompt = (
                    f"<user_requirements>{state['user_requirement']}</user_requirements>{missing_boundary_info}\n"
                    "Please create Python code using the GMSH library to generate a mesh based on the user requirements. "
                    "Use boundary names specified in user requirements (e.g., 'inlet', 'outlet', 'wall', 'cylinder', etc.). "
                    "Return ONLY the complete Python code without any additional text."
                )
                
                # Generate Python code using LLM
                python_response = state["llm_service"].invoke(python_prompt, GMSH_PYTHON_SYSTEM_PROMPT, pydantic_obj=GMSHPythonCode)
                
                if not python_response.python_code:
                    print("Failed to generate GMSH Python code")
                    if gmsh_python_current_loop >= max_loop:
                        return {
                            "mesh_info": None,
                            "mesh_commands": [],
                            "mesh_file_destination": None,
                            "error_logs": error_logs
                        }
                    continue
                
                python_code_to_use = python_response.python_code
                mesh_type = python_response.mesh_type
                geometry_type = python_response.geometry_type
            else:
                # Use corrected code from previous attempt
                mesh_type = "3D"  # Default for corrected code
                geometry_type = "corrected"  # Indicate this is corrected code
            
            # Save the Python file
            save_file(python_file, python_code_to_use)
            print(f"Created GMSH Python file: {python_file}")
            
            # Clear the corrected code flag since we're using it
            corrected_python_code = None
            
            # Run the Python code to generate the mesh
            print("Running GMSH Python code to generate mesh...")
            
            # Use Popen to get real-time output while still capturing stderr for error correction
            process = subprocess.Popen(
                ["python", python_file],
                cwd=case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream stdout in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            
            # Get stderr for potential error correction
            stderr_output = process.stderr.read()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args, stderr=stderr_output)
            
            print("GMSH Python mesh generation completed successfully")
            
            # Verify the mesh file was created
            if not os.path.exists(msh_file):
                print("Error: Mesh file was not created by GMSH Python")
                if stderr_output:
                    print(f"GMSH Python error output: {stderr_output}")
                    # Try to correct the Python code based on the error
                    if gmsh_python_current_loop < max_loop:
                        print("Attempting to correct Python code based on error...")
                        corrected_code = _correct_gmsh_python_code(
                            state, 
                            python_code_to_use, 
                            stderr_output
                        )
                        if corrected_code:
                            corrected_python_code = corrected_code
                            print("Generated corrected Python code for next attempt")
                            continue
                if gmsh_python_current_loop >= max_loop:
                    return {
                        "mesh_info": None,
                        "mesh_commands": [],
                        "mesh_file_destination": None,
                        "error_logs": error_logs
                    }
                continue
            
            print(f"Successfully created mesh file: {msh_file}")
            
            # ========== INTEGRATED PREPROCESSOR OPERATIONS ==========
            print("Starting OpenFOAM conversion and boundary checking...")
            
            try:
                # Create constant and system directories
                constant_dir = os.path.join(case_dir, "constant")
                system_dir = os.path.join(case_dir, "system")
                os.makedirs(constant_dir, exist_ok=True)
                os.makedirs(system_dir, exist_ok=True)
                
                # Create basic controlDict file
                controldict_prompt = (
                    f"<user_requirements>{state['user_requirement']}</user_requirements>\n"
                    "Please create a basic controlDict file for mesh conversion. "
                    "The file should include only the essential settings needed for gmshToFoam to work. "
                    "IMPORTANT: Return ONLY the complete controlDict file content without any additional text."
                )
                
                controldict_content = state["llm_service"].invoke(controldict_prompt, CONTROLDICT_SYSTEM_PROMPT).strip()
                
                if controldict_content:
                    controldict_file = os.path.join(system_dir, "controlDict")
                    save_file(controldict_file, controldict_content)
                    print("Created basic controlDict file for mesh conversion")
                
                # Run gmshToFoam command
                print("Running gmshToFoam conversion...")
                result = subprocess.run(
                    ["gmshToFoam", "geometry.msh"],
                    cwd=case_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"gmshToFoam command output: {result.stdout}")
                
                # Check if the mesh was converted successfully
                polyMesh_dir = os.path.join(constant_dir, "polyMesh")
                if not os.path.exists(polyMesh_dir):
                    raise subprocess.CalledProcessError(1, "gmshToFoam", "polyMesh directory not created")
                
                # Check boundary file for boundaries
                boundary_file = os.path.join(polyMesh_dir, "boundary")
                if os.path.exists(boundary_file):
                    all_present, missing_boundaries, found_boundaries = check_boundary_file_for_missing_boundaries(
                        boundary_file, expected_boundaries
                    )
                    
                    print(f"Found boundaries in OpenFOAM: {found_boundaries}")
                    print(f"Expected boundaries: {expected_boundaries}")
                    
                    # Simple: If sets don't match, ask LLM to fix
                    if set(found_boundaries) != set(expected_boundaries):
                        print(f"Boundary mismatch detected. Found: {found_boundaries}, Expected: {expected_boundaries}")
                        # Store for LLM context
                        state['found_boundaries'] = found_boundaries
                        state['expected_boundaries'] = expected_boundaries
                        # Read current generate_mesh.py code
                        with open(python_file, 'r') as f:
                            current_code = f.read()
                        if gmsh_python_current_loop < max_loop:
                            print("Retrying mesh generation due to boundary mismatch...")
                            boundary_error = (
                                f"Boundary mismatch after gmshToFoam. "
                                f"Found boundaries: {found_boundaries}. "
                                f"Expected boundaries: {expected_boundaries}. "
                                "Please correct the mesh code so that the boundaries in the OpenFOAM boundary file match the expected boundaries exactly."
                            )
                            error_logs.append(boundary_error)
                            corrected_code = _correct_gmsh_python_code(
                                state,
                                current_code,
                                boundary_error
                            )
                            if corrected_code:
                                corrected_python_code = corrected_code
                                print("Generated corrected Python code for next attempt (boundary mismatch)")
                                continue
                        else:
                            print(f"Failed to generate correct boundaries after {max_loop} attempts")
                            return {
                                "mesh_info": None,
                                "mesh_commands": [],
                                "mesh_file_destination": None,
                                "error_logs": error_logs
                            }
                    else:
                        print("All boundaries match expected in OpenFOAM boundary file")
                        # Clear any previous boundary issues
                        if 'found_boundaries' in state:
                            del state['found_boundaries']
                        if 'expected_boundaries' in state:
                            del state['expected_boundaries']
                        missing_boundaries = []
                        
                        # Run checkMesh to verify mesh quality BEFORE boundary file update
                        checkmesh_success, should_continue = run_checkmesh_and_correct(
                            state, case_dir, python_file, max_loop, gmsh_python_current_loop, corrected_python_code, error_logs
                        )
                        
                        if not checkmesh_success:
                            if should_continue:
                                continue  # Continue to next iteration with corrected code
                            else:
                                print(f"Failed to resolve checkMesh issues after {max_loop} attempts")
                                return {
                                    "mesh_info": None,
                                    "mesh_commands": [],
                                    "mesh_file_destination": None,
                                    "error_logs": error_logs
                                }
                        
                        # Handle boundary conditions based on user requirements
                        with open(boundary_file, 'r') as f:
                            boundary_content = f.read()
                        
                        boundary_prompt = (
                            f"<user_requirements>{state['user_requirement']}</user_requirements>\n"
                            f"<boundary_file_content>{boundary_content}</boundary_file_content>\n"
                            "Please analyze the user requirements and boundary file content. "
                            "Identify which boundary is to be modified based on the boundaries mentioned in the user requirements."
                            "If this is a 2D simulation, modify ONLY the appropriate boundary to 'empty' type and 'empty' physicalType. "
                            "Based on the no slip boundaries mentioned in the user requirements, modify the appropriate boundary/boundaries to type 'wall' and physicalType 'wall'. "
                            "If this is a 3D simulation, only modify the appropriate boundary/boundaries to type 'wall' and physicalType 'wall'."
                            "IMPORTANT: Do not change any other boundaries - leave them exactly as they are. "
                            "Return ONLY the complete boundary file content with any necessary modifications. No additional text."
                        )
                        
                        updated_boundary_content = state["llm_service"].invoke(boundary_prompt, BOUNDARY_SYSTEM_PROMPT).strip()
                        
                        if updated_boundary_content:
                            save_file(boundary_file, updated_boundary_content)
                            print("Boundary file updated based on simulation requirements")
                
                # Create .foam file
                foam_file = os.path.join(case_dir, f"{os.path.basename(case_dir)}.foam")
                with open(foam_file, 'w') as f:
                    pass
                
                print("OpenFOAM conversion, boundary setup, and mesh quality check completed successfully")
                
                # Generate mesh commands for the InputWriter node
                mesh_commands = [   # Check mesh quality  # Renumber mesh for better performance
                ]
                
                # Update state with mesh information
                return {
                    "mesh_info": {
                        "mesh_file_path": msh_file,
                        "mesh_file_type": "gmsh",
                        "mesh_description": f"GMSH generated {geometry_type} mesh",
                        "requires_blockmesh_removal": True
                    },
                    "mesh_commands": mesh_commands,
                    "mesh_file_destination": msh_file,
                    "custom_mesh_used": True,
                    "error_logs": error_logs
                }
                
            except subprocess.CalledProcessError as e:
                error_message = f"Error in OpenFOAM conversion: {str(e)}"
                if e.stdout:
                    error_message += f"\nSTDOUT:\n{e.stdout}"
                if e.stderr:
                    error_message += f"\nSTDERR:\n{e.stderr}"
                print(error_message)
                error_logs.append(error_message)
                
                # Retry mesh generation
                if gmsh_python_current_loop < max_loop:
                    print("Retrying mesh generation due to OpenFOAM conversion error...")
                    continue
                else:
                    print(f"Failed to convert mesh to OpenFOAM format after {max_loop} attempts")
                    return {
                        "mesh_info": None,
                        "mesh_commands": [],
                        "mesh_file_destination": None,
                        "error_logs": error_logs
                    }
            
        except subprocess.CalledProcessError as e:
            error_message = f"Error in GMSH Python mesh generation (attempt {gmsh_python_current_loop}): {str(e)}"
            if e.stdout:
                error_message += f"\nSTDOUT:\n{e.stdout}"
            if e.stderr:
                error_message += f"\nSTDERR:\n{e.stderr}"
            print(error_message)
            error_logs.append(error_message)
            
            # Try to correct the Python code based on the error
            if gmsh_python_current_loop < max_loop:
                print("Attempting to correct Python code based on error...")
                try:
                    # Read the current Python file
                    with open(python_file, 'r') as f:
                        current_code = f.read()
                    
                    corrected_code = _correct_gmsh_python_code(
                        state, 
                        current_code, 
                        e.stderr
                    )
                    if corrected_code:
                        corrected_python_code = corrected_code
                        print("Generated corrected Python code for next attempt")
                        continue
                except Exception as correction_error:
                    print(f"Error during Python code correction: {correction_error}")
            
            if gmsh_python_current_loop >= max_loop:
                print(f"Failed to generate mesh after {max_loop} attempts")
                return {
                    "mesh_info": None,
                    "mesh_commands": [],
                    "mesh_file_destination": None,
                    "error_logs": error_logs
                }
            
        except Exception as e:
            print(f"Error in GMSH Python node: {str(e)}")
            if gmsh_python_current_loop >= max_loop:
                return {
                    "mesh_info": None,
                    "mesh_commands": [],
                    "mesh_file_destination": None,
                    "error_logs": error_logs
                }
            continue
    
    return {
        "mesh_info": None,
        "mesh_commands": [],
        "mesh_file_destination": None,
        "error_logs": error_logs
    }

def handle_standard_mesh(state, case_dir):
    """Handle standard OpenFOAM mesh generation."""
    print("============================== Standard Mesh Generation ==============================")
    print("Using standard OpenFOAM mesh generation (blockMesh, snappyHexMesh, etc.)")
    return {
        "mesh_info": None,
        "mesh_commands": [],
        "mesh_file_destination": None,
        "custom_mesh_used": False,
        "error_logs": []
    }

def meshing_node(state):
    """
    Meshing node: Handle different mesh scenarios based on user requirements.
    
    Three scenarios:
    1. Custom mesh: User provides existing mesh file (uses preprocessor logic)
    2. GMSH mesh: User wants mesh generated using GMSH (uses gmsh python logic)
    3. Standard mesh: User wants standard OpenFOAM mesh generation (returns None)
    
    Updates state with:
      - mesh_info: Information about the custom mesh
      - mesh_commands: Commands needed for mesh processing
      - mesh_file_destination: Where the mesh file should be placed
    """
    # 1. 读取state中的配置信息、用户需求和算例目录
    config = state["config"]
    user_requirement = state["user_requirement"]
    case_dir = state["case_dir"]
    print(f"[meshing_node] config: {config}")
    print(f"[meshing_node] user_requirement: {user_requirement}")
    print(f"[meshing_node] case_dir: {case_dir}")
    
    # Get mesh type from state (determined by router)
    mesh_type = state.get("mesh_type", "standard_mesh")
    
    # Handle mesh based on type determined by router
    if mesh_type == "custom_mesh":
        print("Router determined: Custom mesh requested.")
        return handle_custom_mesh(state, case_dir)
    elif mesh_type == "gmsh_mesh":
        print("Router determined: GMSH mesh requested.")
        return handle_gmsh_mesh(state, case_dir)
    else:
        print("Router determined: Standard mesh generation.")
        return handle_standard_mesh(state, case_dir)
