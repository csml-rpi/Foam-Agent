from services.mesh import copy_custom_mesh, prepare_standard_mesh, handle_gmsh_mesh as service_handle_gmsh_mesh

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
    config = state["config"]
    user_requirement = state["user_requirement"]
    case_dir = state["case_dir"]
    
    # Get mesh type from state (determined by router)
    mesh_type = state.get("mesh_type", "standard_mesh")
    
    # Handle mesh based on type determined by router
    if mesh_type == "custom_mesh":
        print("Router determined: Custom mesh requested.")
        return copy_custom_mesh(state.get("custom_mesh_path"), user_requirement, case_dir)  # service
    elif mesh_type == "gmsh_mesh":
        print("Router determined: GMSH mesh requested.")
        return service_handle_gmsh_mesh(state, case_dir)  # service
    else:
        print("Router determined: Standard mesh generation.")
        return prepare_standard_mesh(user_requirement, case_dir)  # service
