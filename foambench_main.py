import os
import subprocess
import sys
import argparse
import shlex

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Workflow Interface")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        '--openfoam_path',
        type=str,
        required=False,
        help="Path to OpenFOAM installation (WM_PROJECT_DIR)"
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=os.path.join(base_dir, "output"),
        help="Base output directory for benchmark results (default: <dir_of_foambench_main.py>/output)"
    )
    parser.add_argument(
        '--prompt_path',
        type=str,
        required=False,
        default=os.path.join(base_dir, "user_requirement.txt"),
        help="User requirement file path for the benchmark (default: <dir_of_foambench_main.py>/user_requirement.txt)"
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

    # Create the output folder
    os.makedirs(args.output, exist_ok=True)

    # Build main workflow command with optional custom mesh path
    main_cmd = f"{sys.executable} -m foamagent.main --prompt_path='{args.prompt_path}' --output_dir='{args.output}'"
    if args.custom_mesh_path:
        main_cmd += f" --custom_mesh_path='{args.custom_mesh_path}'"
    
    print(f"Main workflow command: {main_cmd}")
    
    print("Starting workflow...")
    run_command(main_cmd)
    print("Workflow command finished.")

if __name__ == "__main__":
    # Examples (paths are resolved relative to the directory containing this file):
    #   python foambench_main.py
    #   python foambench_main.py --output output --prompt_path user_requirement.txt
    #   python foambench_main.py --output output --prompt_path user_requirement.txt --custom_mesh_path my_mesh.msh
    main()
