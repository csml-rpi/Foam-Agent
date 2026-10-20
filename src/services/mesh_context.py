"""Mesh metadata shared by the Meshing backends."""

import re
from pathlib import Path


def mesh_state(case_root):
    root = Path(case_root)
    if (root / "constant" / "polyMesh").is_dir():
        return "existing-polyMesh"
    if (root / "system" / "blockMeshDict").is_file():
        return "blockMesh"
    return "missing"


def is_mesh_file(relative):
    path = Path(relative)
    name = path.name
    return (
        "polyMesh" in path.parts or "triSurface" in path.parts
        or name in {"Allmesh", "generate_mesh.py", "topoSetDict", "createPatchDict",
                    "surfaceFeatureExtractDict", "surfaceFeaturesDict", "extrudeMeshDict"}
        or "mesh" in name.lower() and name.endswith("Dict")
        or path.suffix.lower() in {".msh", ".geo", ".stl", ".obj"}
    )


def mesh_info(case_dir, previous=None):
    root = Path(case_dir) / "constant" / "polyMesh"
    boundary = root / "boundary"
    patches = []
    if boundary.is_file():
        text = boundary.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)
        for name, body in re.findall(r"([\w.:-]+)\s*\{([^{}]*)\}", text):
            kind = re.search(r"\btype\s+(\w+)\s*;", body)
            count = re.search(r"\bnFaces\s+(\d+)\s*;", body)
            if kind and count:
                patches.append({"name": name, "type": kind[1], "n_faces": int(count[1])})
    return {**(previous or {}), "poly_mesh_path": str(root), "patches": patches}
