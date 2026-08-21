"""Command policy shared by OpenFOAM generation, import, and execution.

The list intentionally describes commands that can create or mutate a mesh.
Every such command must be followed by ``checkMesh`` before a flow solver is
allowed to start.
"""

from __future__ import annotations


MESH_MUTATING_COMMANDS = frozenset(
    {
        "autoPatch",
        "blockMesh",
        "cartesianMesh",
        "cfx4ToFoam",
        "createBaffles",
        "createNonConformalCouples",
        "createPatch",
        "extrudeMesh",
        "fluentMeshToFoam",
        "foamyMesh",
        "gmshToFoam",
        "ideasUnvToFoam",
        "mergeMeshes",
        "netgenNeutralToFoam",
        "refineHexMesh",
        "refineMesh",
        "renumberMesh",
        "snappyHexMesh",
        "splitBaffles",
        "splitMeshRegions",
        "star4ToFoam",
        "stitchMesh",
        "tetgenToFoam",
        "topoSet",
        "transformPoints",
    }
)
