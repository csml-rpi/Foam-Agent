from foamagent.services.mesh import copy_custom_mesh, prepare_standard_mesh, handle_gmsh_mesh as service_handle_gmsh_mesh

from foamagent.logger import get_logger

logger = get_logger(__name__)

def handle_standard_mesh(state, case_dir):
    """Handle standard OpenFOAM mesh generation."""
    logger.info("<meshing type=\"standard\">")
    logger.info("Using standard OpenFOAM mesh generation (blockMesh, snappyHexMesh, etc.)")
    logger.info("</meshing>")
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
    config = state["config"]
    user_requirement = state["user_requirement"]
    case_dir = state["case_dir"]
    
    # Get mesh type from state (determined by router)
    mesh_type = state.get("mesh_type", "standard_mesh")
    
    # Handle mesh based on type determined by router
    logger.info("<meshing>")
    if mesh_type == "custom_mesh":
        logger.info("<mesh_routing>Custom mesh requested.</mesh_routing>")
        result = copy_custom_mesh(state.get("custom_mesh_path"), user_requirement, case_dir)  # service
    elif mesh_type == "gmsh_mesh":
        logger.info("<mesh_routing>GMSH mesh requested.</mesh_routing>")
        result = service_handle_gmsh_mesh(state, case_dir)  # service
    else:
        logger.info("<mesh_routing>Standard mesh generation.</mesh_routing>")
        result = prepare_standard_mesh(user_requirement, case_dir)  # service
    logger.info("</meshing>")
    return result
