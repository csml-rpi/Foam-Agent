import os
import shutil
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import save_file, remove_file

# MeshInfoPydantic用于描述自定义网格文件的信息
class MeshInfoPydantic(BaseModel):
    mesh_file_path: str = Field(description="Path to the custom mesh file (e.g., .msh, .stl, .obj)")
    mesh_file_type: str = Field(description="Type of mesh file (gmsh, stl, obj, etc.)")
    mesh_description: str = Field(description="Description of the mesh and any specific requirements")
    requires_blockmesh_removal: bool = Field(description="Whether to remove blockMeshDict file", default=True)

# MeshCommandsPydantic用于描述处理网格所需的OpenFOAM命令及目标路径
class MeshCommandsPydantic(BaseModel):
    mesh_commands: List[str] = Field(description="List of OpenFOAM commands needed to process the custom mesh")
    mesh_file_destination: str = Field(description="Destination path for the mesh file in the case directory")


def meshing_node(state):
    """
    Meshing node: Handle custom mesh files provided by the user.
    Processes mesh files like Gmsh (.msh), STL, OBJ, etc. using OpenFOAM tools.
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
    
    # 2. 检查用户需求中是否包含自定义网格信息
    # 这里是mock实现，实际应由LLM解析用户输入
    
    # mesh_system_prompt和mesh_user_prompt用于LLM解析用户需求
    mesh_system_prompt = (
        "You are an expert in OpenFOAM mesh processing. "
        "Analyze the user requirement to identify if they want to use a custom mesh file. "
        "If a custom mesh is mentioned, extract the mesh file path, type, and description. "
        "Return the information in a structured format."
    )
    
    mesh_user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Identify if the user wants to use a custom mesh file. "
        "If yes, extract the mesh file path, type (gmsh, stl, obj, etc.), and description. "
        "If no custom mesh is mentioned, return None for all fields."
    )
    
    # 3. 通过LLM服务解析用户需求，获得网格信息（此处为mock）
    mesh_response = state["llm_service"].invoke(mesh_user_prompt, mesh_system_prompt, pydantic_obj=MeshInfoPydantic)
    print(f"[meshing_node] mesh_response: {mesh_response}")
    
    # 4. 这里直接mock一个网格信息，实际应为上一步的解析结果
    mesh_info = {
        "mesh_file_path": "/path/to/user/mesh.msh",  # Mock path
        "mesh_file_type": "gmsh",
        "mesh_description": "Custom mesh for complex geometry with refined regions",
        "requires_blockmesh_removal": True
    }
    print(f"[meshing_node] mesh_info: {mesh_info}")
    
    # 5. 如果没有自定义网格，直接跳过该节点
    if not mesh_info["mesh_file_path"] or mesh_info["mesh_file_path"] == "None":
        print("No custom mesh requested. Skipping meshing node.")
        return {
            **state,
            "mesh_info": None,
            "mesh_commands": [],
            "mesh_file_destination": None
        }
    
    print(f"Processing custom mesh: {mesh_info['mesh_file_path']}")
    print(f"Mesh type: {mesh_info['mesh_file_type']}")
    print(f"Description: {mesh_info['mesh_description']}")
    
    # 6. 生成网格文件在算例目录下的目标路径
    mesh_file_destination = os.path.join(case_dir, "constant", "triSurface", os.path.basename(mesh_info["mesh_file_path"]))
    print(f"[meshing_node] mesh_file_destination: {mesh_file_destination}")
    
    # 7. 根据网格类型生成OpenFOAM处理命令
    mesh_commands = []
    
    if mesh_info["mesh_file_type"].lower() == "gmsh":
        # Gmsh类型网格需先转换格式
        mesh_commands = [
            "gmshToFoam",  # 转换.msh为OpenFOAM格式
            "checkMesh",   # 检查网格质量
            "renumberMesh -overwrite"  # 重新编号提升性能
        ]
    elif mesh_info["mesh_file_type"].lower() == "stl":
        # STL类型网格
        mesh_commands = [
            "surfaceMeshTriangulate",  # 三角化表面网格
            "checkMesh",               # 检查网格质量
        ]
    elif mesh_info["mesh_file_type"].lower() == "obj":
        # OBJ类型需先转为STL
        mesh_commands = [
            "objToSTL",                # OBJ转STL
            "surfaceMeshTriangulate",  # 三角化表面网格
            "checkMesh",               # 检查网格质量
        ]
    else:
        # 其他类型，默认只检查网格
        mesh_commands = [
            "checkMesh",  # 检查网格质量
        ]
    print(f"[meshing_node] mesh_commands: {mesh_commands}")
    
    # 8. 如果使用自定义网格，需要移除blockMeshDict
    if mesh_info["requires_blockmesh_removal"]:
        blockmesh_path = os.path.join(case_dir, "system", "blockMeshDict")
        print(f"[meshing_node] blockmesh_path: {blockmesh_path}")
        if os.path.exists(blockmesh_path):
            print(f"Removing blockMeshDict: {blockmesh_path}")
            remove_file(blockmesh_path)
    
    # 9. 生成MeshCommandsPydantic对象，供后续节点使用
    mesh_commands_pydantic = MeshCommandsPydantic(
        mesh_commands=mesh_commands,
        mesh_file_destination=mesh_file_destination
    )
    print(f"[meshing_node] mesh_commands_pydantic: {mesh_commands_pydantic}")
    
    # 10. 创建目标目录（如不存在）
    os.makedirs(os.path.dirname(mesh_file_destination), exist_ok=True)
    print(f"[meshing_node] Created directory: {os.path.dirname(mesh_file_destination)}")
    
    # 11. 拷贝网格文件到算例目录（此处为mock，实际应拷贝真实文件）
    print(f"Mesh file would be copied to: {mesh_file_destination}")
    
    # 12. 更新state，返回处理结果
    return {
        **state,
        "mesh_info": mesh_info,
        "mesh_commands": mesh_commands,
        "mesh_file_destination": mesh_file_destination,
        "custom_mesh_used": True
    }
