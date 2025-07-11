import os
import subprocess
import sys
import argparse
import shlex

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Workflow Interface")
    parser.add_argument(
        '--openfoam_path',
        type=str,
        required=True,
        help="Path to OpenFOAM installation (WM_PROJECT_DIR)"
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Base output directory for benchmark results"
    )
    parser.add_argument(
        '--prompt_path',
        type=str,
        required=True,
        help="User requirement file path for the benchmark"
    )
    parser.add_argument(
        '--custom_mesh_path',
        type=str,
        default=None,
        help="Path to custom mesh file (e.g., .msh, .stl, .obj). If not provided, no custom mesh will be used."
    )
    return parser.parse_args()

def run_command(command_str):
    """
    Execute a command string using the current terminal's input/output,
    with the working directory set to the directory of the current file.
    
    Parameters:
        command_str (str): The command to execute, e.g. "python main.py --output_dir xxxx" 
                           or "bash xxxxx.sh".
    """
    # Split the command string into a list of arguments
    args = shlex.split(command_str)
    # Set the working directory to the directory of the current file
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=sys.stdin
        )
        print(f"Finished command: Return Code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(e.returncode)

def main():
    args = parse_args()
    print(args)

    # Set environment variables
    WM_PROJECT_DIR = args.openfoam_path
    # Check if OPENAI_API_KEY is available in the environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY is not set in the environment.")
        sys.exit(1)

    # Create the output folder
    os.makedirs(args.output, exist_ok=True)

    # Define the list of scripts to be executed.
    # Each tuple consists of (script_path, list_of_arguments).
    # Scripts can be Python or shell scripts.
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"script_dir: {script_dir}")

    SCRIPTS = []
    
    # Preprocess the OpenFOAM tutorials    
    if not os.path.exists(f"{script_dir}/database/raw/openfoam_tutorials_details.txt"):
        SCRIPTS.append(f"python database/script/tutorial_parser.py --output_dir=./database/raw --wm_project_dir={WM_PROJECT_DIR}")
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_command_help"):
        SCRIPTS.append(f"python database/script/faiss_command_help.py --database_path=./database")
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_allrun_scripts"):
        SCRIPTS.append(f"python database/script/faiss_allrun_scripts.py --database_path=./database")
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_tutorials_structure"):
        SCRIPTS.append(f"python database/script/faiss_tutorials_structure.py --database_path=./database")
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_tutorials_details"):
        SCRIPTS.append(f"python database/script/faiss_tutorials_details.py --database_path=./database")
    
    # Build main workflow command with optional custom mesh path
    main_cmd = f"python src/main.py --prompt_path='{args.prompt_path}' --output_dir='{args.output}'"
    if args.custom_mesh_path:
        main_cmd += f" --custom_mesh_path='{args.custom_mesh_path}'"
    
    print(f"Main workflow command: {main_cmd}")
    # Main workflow
    SCRIPTS.extend([
        main_cmd
    ])

    print("Starting workflow...")
    for script in SCRIPTS:
        run_command(script)
    print("Workflow completed successfully.")

if __name__ == "__main__":
    ## python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path "./user_requirement.txt"
    ## python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path "./user_requirement.txt" --custom_mesh_path "./my_mesh.msh"
    main()
