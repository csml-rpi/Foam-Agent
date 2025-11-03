#!/usr/bin/env python3
"""Test lid-driven cavity case using services directly.

This script tests the OpenFOAM case generation for lid-driven cavity flow
by calling services functions directly, without using MCP server.
"""

import os
import sys
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.plan import (
    parse_requirement_to_case_info,
    resolve_case_dir,
    retrieve_references,
    generate_simulation_plan
)
from services.input_writer import initial_write
from services.mesh import prepare_standard_mesh
from services.run_local import run_allrun_and_collect_errors
from services.review import review_error_logs
from services.visualization import (
    ensure_foam_file,
    generate_pyvista_script,
    run_pyvista_script
)
from config import Config


def main():
    """Run lid-driven cavity test using services."""
    
    print("🚀 Lid-Driven Cavity Test (Services)")
    print("=" * 60)
    
    # Configuration
    config = Config()
    
    # User requirement for lid-driven cavity
    user_requirement = """
    Do an incompressible lid driven cavity flow. 
    The cavity is a square with dimensions normalized to 1 unit on both the x and y axes and very thin in the z-direction (0.1 unit scaled down by a factor of 0.1, making it effectively 2D). 
    Use a grid of 20X20 in x and y direction and 1 cell in z-direction(due to the expected 2D flow characteristics). 
    The top wall ('movingWall') moves in the x-direction with a uniform velocity of 1 m/s. 
    The 'fixedWalls' have a no-slip boundary condition (velocity equal to zero at the wall). 
    The front and back faces are designated as 'empty'. 
    The simulation runs from time 0 to 10 with a time step of 0.005 units, and results are output every 100 time steps. 
    The viscosity (`nu`) is set as constant with a value of 1e-05 m^2/s
    """
    
    results = {}
    
    try:
        # Step 1: Create case
        print("\n📁 Step 1: Creating case")
        print("-" * 40)
        
        # Load case statistics
        case_stats_path = os.path.join(config.database_path, "raw", "openfoam_case_stats.json")
        with open(case_stats_path, 'r') as f:
            case_stats = json.load(f)
        
        # Parse requirements to case info
        case_info = parse_requirement_to_case_info(user_requirement, case_stats)
        
        # Resolve case directory
        case_name = "lid_driven_cavity_20x20"
        case_dir = resolve_case_dir(
            case_name=case_name,
            case_dir="./lid_driven_cavity_test",
            run_times=config.run_times
        )
        
        # Create case directory
        os.makedirs(case_dir, exist_ok=True)
        case_id = os.path.basename(case_dir)
        
        print(f"✅ Created case: {case_id}")
        print(f"   Directory: {case_dir}")
        print(f"   Case info: {case_info}")
        results['case_creation'] = True
        
        # Step 2: Plan simulation
        print("\n📋 Step 2: Planning simulation")
        print("-" * 40)
        
        plan_data = generate_simulation_plan(
            user_requirement=user_requirement,
            case_stats=case_stats,
            case_dir=case_dir,
            searchdocs=config.searchdocs,
            file_dependency_threshold=config.file_dependency_threshold
        )
        
        print(f"✅ Generated {len(plan_data['subtasks'])} subtasks")
        for i, subtask in enumerate(plan_data['subtasks']):
            print(f"   {i+1}. {subtask['file_name']} in {subtask['folder_name']}")
        results['planning'] = True
        
        # Step 3: Generate files
        print("\n📝 Step 3: Generating OpenFOAM files")
        print("-" * 40)
        
        # Use subtasks directly from plan_data
        subtasks = plan_data["subtasks"]
        
        # Use references from plan_data
        tutorial_reference = plan_data["tutorial_reference"]
        dir_structure = plan_data["dir_structure_reference"]
        allrun_reference = plan_data["allrun_reference"]
        file_dependency_flag = plan_data["file_dependency_flag"]
        
        # Generate files
        case_info_str = f"case name: {plan_data['case_name']}\ncase domain: {plan_data['case_domain']}\ncase category: {plan_data['case_category']}\ncase solver: {plan_data['case_solver']}"
        result = initial_write(
            case_dir=case_dir,
            subtasks=subtasks,
            user_requirement=user_requirement,
            tutorial_reference=tutorial_reference,
            case_solver=plan_data["case_solver"],
            file_dependency_flag=file_dependency_flag,
            case_info=case_info_str,
            allrun_reference=allrun_reference,
            database_path=str(config.database_path),
            searchdocs=config.searchdocs
        )
        
        # Count generated files from dir_structure
        dir_structure = result.get("dir_structure", {})
        file_count = sum(len(files) for files in dir_structure.values())
        print(f"✅ Generated {file_count} files in {len(dir_structure)} directories")
        results['file_generation'] = True
        
        # Step 4: Prepare mesh
        print("\n🕸️ Step 4: Preparing mesh")
        print("-" * 40)
        
        prepare_standard_mesh(user_requirement, case_dir)
        
        mesh_dir = os.path.join(case_dir, 'constant', 'polyMesh')
        if os.path.exists(mesh_dir):
            print(f"✅ Mesh directory ready: {mesh_dir}")
        else:
            print(f"⚠️ Mesh directory will be created during Allrun execution")
        results['mesh_preparation'] = True
        
        # Step 5: Run simulation
        print("\n🏃 Step 5: Running simulation")
        print("-" * 40)
        
        errors = run_allrun_and_collect_errors(
            case_dir=case_dir,
            timeout=600,  # 10 minutes
            max_retries=3
        )
        
        # Get logs
        logs = {}
        out_path = os.path.join(case_dir, 'Allrun.out')
        err_path = os.path.join(case_dir, 'Allrun.err')
        
        if os.path.exists(out_path):
            with open(out_path, 'r', errors='ignore') as f:
                logs['Allrun.out'] = f.read()
        
        if os.path.exists(err_path):
            with open(err_path, 'r', errors='ignore') as f:
                logs['Allrun.err'] = f.read()
        
        status = "completed" if not errors else "failed"
        print(f"✅ Simulation {status}")
        
        if errors:
            print(f"   Errors found: {len(errors)}")
            for error in errors[:3]:
                print(f"   - {error}")
        else:
            print(f"   No errors detected")
        
        results['simulation_run'] = (status == 'completed')
        
        # Step 6: Review results
        print("\n🔍 Step 6: Reviewing results")
        print("-" * 40)
        
        # Convert errors to list of strings if needed
        error_logs = errors if isinstance(errors, list) else [str(err) for err in errors]
        
        review_content, _ = review_error_logs(
            tutorial_reference=tutorial_reference,
            foamfiles=None,
            error_logs=error_logs,
            user_requirement=user_requirement,
            history_text=None
        )
        
        print(f"✅ Review completed")
        print(f"   Analysis: {len(review_content)} characters")
        results['review'] = True
        
        # Step 7: Generate visualization
        print("\n📊 Step 7: Generating visualization")
        print("-" * 40)
        
        foam_file = ensure_foam_file(case_dir)
        script = generate_pyvista_script(
            case_dir=case_dir,
            foam_file=foam_file,
            user_requirement="velocity field",
            previous_errors=[]
        )
        
        ok, img, errs = run_pyvista_script(case_dir, script)
        
        if ok and img:
            print(f"✅ Generated visualization: {img}")
        else:
            print(f"⚠️ Visualization issues: {errs}")
        results['visualization'] = ok
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:25} {status}")
        
        print(f"\nOverall: {passed}/{total} steps passed")
        
        if passed == total:
            print("🎉 All steps completed successfully!")
            return 0
        else:
            print(f"⚠️ {total - passed} steps had issues")
            return 1
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

