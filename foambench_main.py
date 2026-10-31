import os
import subprocess
import sys
import argparse
import shlex

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Workflow Interface")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        '--openfoam_target',
        type=str,
        default=None,
        help="Explicit native target, e.g. esi-v2006. Omit to preserve existing routing."
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
        default=None,
        help="User requirement file path for the benchmark (default: <dir_of_foambench_main.py>/user_requirement.txt)"
    )
    parser.add_argument(
        '--case_path',
        type=str,
        default=None,
        help=(
            "Existing OpenFOAM case directory or ZIP archive. Uses the explicitly "
            "configured target; otherwise attempts to detect Foundation v10 or ESI "
            "v2006 from case headers. Specify --openfoam_target if detection is inconclusive."
        )
    )
    parser.add_argument(
        '--case_subdir',
        type=str,
        default=None,
        help="Relative case directory inside --case_path when multiple cases are present."
    )
    parser.add_argument(
        '--custom_mesh_path',
        type=str,
        default=None,
        help="Path to a Gmsh .msh file (ASCII 2.2 format). If not provided, no custom mesh will be used."
    )
    parser.add_argument(
        '--overwrite_output',
        action='store_true',
        help="Explicitly allow replacing an existing non-empty output directory."
    )
    args = parser.parse_args()
    if args.case_subdir and not args.case_path:
        parser.error("--case_subdir requires --case_path.")
    return args

def run_command(command):
    """
    Execute a command string using the current terminal's input/output,
    with the working directory set to the directory of the current file.
    
    Parameters:
        command: A command string or an argument sequence, e.g.
                 ``["python", "main.py", "--output_dir", "xxxx"]``.
    """
    # Preserve argument boundaries for paths containing spaces.  Accepting a
    # string retains compatibility with callers outside this entry point.
    command_args = shlex.split(command) if isinstance(command, str) else list(command)
    # Set the working directory to the directory of the current file
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    try:
        result = subprocess.run(
            command_args,
            cwd=cwd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=sys.stdin,
        )
        print(f"Finished command: Return Code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(e.returncode)

def main():
    args = parse_args()
    print(args)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Output creation and any overwrite decision are owned by the workflow's
    # ownership checks before the workflow gets a chance to validate it.

    # Build the workflow invocation as argument tokens, not shell text.
    main_cmd = [sys.executable, "src/main.py", "--output_dir", args.output]
    if args.openfoam_target:
        main_cmd.extend(["--openfoam_target", args.openfoam_target])
    if args.case_path:
        main_cmd.extend(["--case_path", args.case_path])
        if args.prompt_path:
            main_cmd.extend(["--prompt_path", args.prompt_path])
        if args.case_subdir:
            main_cmd.extend(["--case_subdir", args.case_subdir])
    else:
        prompt_path = args.prompt_path or os.path.join(base_dir, "user_requirement.txt")
        main_cmd.extend(["--prompt_path", prompt_path])
    if args.custom_mesh_path:
        main_cmd.extend(["--custom_mesh_path", args.custom_mesh_path])
    if args.overwrite_output:
        main_cmd.append("--overwrite_output")
    
    print(f"Main workflow command: {shlex.join(main_cmd)}")
    
    print("Starting workflow...")
    run_command(main_cmd)
    print("Workflow command finished.")

if __name__ == "__main__":
    # Examples (paths are resolved relative to the directory containing this file):
    #   python foambench_main.py
    #   python foambench_main.py --output output --prompt_path user_requirement.txt
    #   python foambench_main.py --output output --prompt_path user_requirement.txt --custom_mesh_path my_mesh.msh
    main()
