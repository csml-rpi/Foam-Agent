from services.mesh import copy_custom_mesh, prepare_standard_mesh, handle_gmsh_mesh as service_handle_gmsh_mesh
from openfoam_target import runtime_openfoam_target
from pathlib import Path
from utils import load_case_files, FoamfilePydantic

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
    target_version = runtime_openfoam_target(state["config"])
    repairing = bool(state.get("repairing_mesh"))
    repair_kwargs = {
        "repair_feedback": state.get("review_analysis") or "",
        "retry": True,
    } if repairing else {}
    
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
            openfoam_target=target_version,
            **repair_kwargs,
        )
    elif mesh_type == "gmsh_mesh":
        print("<mesh_routing>GMSH mesh requested.</mesh_routing>")
        result = service_handle_gmsh_mesh(
            user_requirement,
            case_dir,
            state["config"].max_loop,
            llm_service=llm_service,
            openfoam_target=target_version,
            **repair_kwargs,
        )
    else:
        print("<mesh_routing>Standard mesh generation.</mesh_routing>")
        result = prepare_standard_mesh()  # service
    print("</meshing>")
    result.update(load_case_files({**state, **result}))
    if result.get("error_logs"):
        script = Path(case_dir) / "generate_mesh.py"
        if script.is_file() and result.get("foamfiles") is not None:
            result["foamfiles"].list_foamfile.append(FoamfilePydantic(
                folder_name="", file_name="generate_mesh.py",
                content=script.read_text(encoding="utf-8"),
            ))
        result["repairing_mesh"] = True
        if not repairing:
            result["mesh_resume_state"] = {
                key: state.get(key) for key in
                ("input_writer_mode", "rewrite_plan", "review_analysis")
            }
    elif repairing:
        result.update(state.get("mesh_resume_state") or {})
        result["repairing_mesh"] = False
        result["mesh_resume_state"] = None
    return result
