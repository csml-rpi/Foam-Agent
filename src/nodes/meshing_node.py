from services.mesh import copy_custom_mesh, prepare_standard_mesh, handle_gmsh_mesh as service_handle_gmsh_mesh

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
    user_requirement = state["user_requirement"]
    case_dir = state["case_dir"]
    llm_service = state.get("llm_service")
    
    # Get mesh type from state (determined by router)
    mesh_type = state.get("mesh_type", "standard_mesh")
    
    # Handle mesh based on type determined by router
    print("<meshing>")
    if mesh_type == "custom_mesh":
        print("<mesh_routing>Custom mesh requested.</mesh_routing>")
        result = copy_custom_mesh(
            state.get("custom_mesh_path"),
            user_requirement,
            case_dir,
            llm_service=llm_service,
        )
    elif mesh_type == "gmsh_mesh":
        print("<mesh_routing>GMSH mesh requested.</mesh_routing>")
        result = service_handle_gmsh_mesh(
            user_requirement,
            case_dir,
            state["config"].max_loop,
            llm_service=llm_service,
        )
    else:
        print("<mesh_routing>Standard mesh generation.</mesh_routing>")
        result = prepare_standard_mesh(user_requirement, case_dir)  # service
    print("</meshing>")
    if result.get("error_logs"):
        # There is no valid case to write or run after mesh preparation fails.
        # Persist a terminal reason so the graph can stop before Input Writer
        # obscures the original mesh error with unrelated dictionary failures.
        result["termination_reason"] = "mesh_generation_failed"
    return result
