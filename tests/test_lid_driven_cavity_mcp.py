#!/usr/bin/env python3
"""Test lid-driven cavity case using MCP server.

This script tests the OpenFOAM case generation for lid-driven cavity flow
through the MCP server interface.
"""

import asyncio
import sys
from pathlib import Path

# Add repository root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import Client


async def main():
    """Run lid-driven cavity test through MCP."""
    
    print("🚀 Lid-Driven Cavity Test (MCP)")
    print("=" * 60)
    
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
        # Connect to MCP server
        print("\n🔌 Connecting to MCP server...")
        client = Client("http://localhost:8080/mcp")
        
        async with client:
            print("✅ Connected to MCP server")
            
            # Step 1: Plan simulation
            print("\n📋 Step 1: Planning simulation")
            print("-" * 40)
            
            plan_response = await client.call_tool(
                "plan",
                {
                    "request": {
                        "user_requirement": user_requirement
                    }
                }
            )
            
            plan_response = plan_response.structured_content or plan_response.data or {}
            print(f"✅ Generated {len(plan_response['subtasks'])} subtasks")
            print(f"   Case name: {plan_response['case_name']}")
            print(f"   Solver: {plan_response['case_solver']}")
            print(f"   Domain: {plan_response['case_domain']}")
            print(f"   Category: {plan_response['case_category']}")
            print(plan_response)
            for i, subtask in enumerate(plan_response['subtasks']):
                print(f"   {i+1}. {subtask['file']} in {subtask['folder']}")
            results['planning'] = True
            
            # Step 2: Generate files
            print("\n📝 Step 2: Generating OpenFOAM files")
            print("-" * 40)
            
            files_response = await client.call_tool(
                "input_writer",
                {
                    "request": {
                        "case_name": plan_response['case_name'],
                        "subtasks": plan_response['subtasks'],
                        "user_requirement": user_requirement,
                        "case_solver": plan_response['case_solver'],
                        "case_domain": plan_response['case_domain'],
                        "case_category": plan_response['case_category']
                    }
                }
            )
            
            files_response = files_response.structured_content or files_response.data or {}
            foamfiles = files_response.get('foamfiles', {})
            num_files = len(foamfiles.get('list_foamfile', [])) if isinstance(foamfiles, dict) else 0
            print(f"✅ Generated {num_files} files")
            case_dir = files_response['case_dir']
            print(f"   Case directory: {case_dir}")
            results['file_generation'] = True

            print(files_response)
            
            # Step 3: Run simulation with error fixing loop
            print("\n🏃 Step 3: Running simulation (with error fixing loop)")
            print("-" * 40)
            
            max_iterations = 5  # Maximum number of fix attempts
            iteration = 0
            run_response = None
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n🔄 Iteration {iteration}/{max_iterations}")
                print(f"Starting simulation in: {case_dir}")
                
                run_response = await client.call_tool(
                    "run",
                    {
                        "request": {
                            "case_dir": case_dir,
                            "timeout": 600  # 10 minutes
                        }
                    }
                )
                
                run_response = run_response.structured_content or run_response.data or {}
                status = run_response['status']
                
                print(run_response)
                print(f"✅ Simulation {status}")
                
                if not run_response.get('errors'):
                    print(f"   No errors detected - simulation completed successfully!")
                    results['simulation_run'] = True
                    results['review'] = True
                    break
                
                # Errors found - review and fix
                print(f"   Errors found: {len(run_response['errors'])}")
                for error in run_response['errors'][:3]:
                    print(f"   - {error}")
                
                # Step 4: Review errors
                print(f"\n🔍 Reviewing errors (iteration {iteration})")
                print("-" * 40)
                
                review_response = await client.call_tool(
                    "review",
                    {
                        "request": {
                            "case_dir": case_dir,
                            "errors": run_response['errors'],
                            "user_requirement": user_requirement
                        }
                    }
                )
                
                review_response = review_response.structured_content or review_response.data or {}
                print(f"✅ Review completed")
                analysis = review_response.get('analysis', '')
                if analysis:
                    print(f"   Analysis: {len(analysis)} characters")
                    # Print first 200 characters of analysis
                    if len(analysis) > 200:
                        print(f"   Preview: {analysis[:200]}...")
                    else:
                        print(f"   Content: {analysis}")
                
                # Step 5: Apply fixes
                print(f"\n🔧 Applying fixes (iteration {iteration})")
                print("-" * 40)
                
                fix_response = await client.call_tool(
                    "apply_fixes",
                    {
                        "request": {
                            "case_dir": case_dir,
                            "error_logs": run_response['errors'],
                            "review_analysis": analysis,
                            "user_requirement": user_requirement
                        }
                    }
                )
                
                fix_response = fix_response.structured_content or fix_response.data or {}
                print(f"✅ Fixes applied")
                updated_files = fix_response.get('updated_files', [])
                fix_status = fix_response.get('status', 'unknown')
                print(f"   Status: {fix_status}")
                print(f"   Updated {len(updated_files)} file(s)")
                if updated_files:
                    for file_path in updated_files[:5]:  # Show first 5 files
                        print(f"   - {file_path}")
                    if len(updated_files) > 5:
                        print(f"   ... and {len(updated_files) - 5} more")
                
                # Will continue loop to run simulation again
            else:
                # Loop completed without success
                print(f"\n⚠️ Maximum iterations ({max_iterations}) reached")
                print(f"   Simulation still has errors after {max_iterations} attempts")
                results['simulation_run'] = False
                results['review'] = True
            
            # Step 6: Generate visualization (only if simulation succeeded)
            if results.get('simulation_run'):
                print("\n📊 Step 6: Generating visualization")
                print("-" * 40)
                
                viz_response = await client.call_tool(
                    "visualization",
                    {
                        "request": {
                            "case_dir": case_dir,
                            "quantity": "velocity",
                            "visualization_type": "pyvista"
                        }
                    }
                )
                
                viz_response = viz_response.structured_content or viz_response.data or {}
                print(viz_response)
                print(f"✅ Generated {len(viz_response.get('artifacts', []))} visualization artifacts")
                results['visualization'] = True
            else:
                print("\n📊 Step 6: Skipping visualization (simulation did not succeed)")
                results['visualization'] = False
        
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
        print(f"\n❌ Test failed: {str(e)}")
        print("\n💡 Make sure the MCP server is running:")
        print("   python -m src.mcp.fastmcp_server --transport http --port 8080")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

