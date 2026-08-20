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
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--prompt_path',
        type=str,
        required=False,
        default=None,
        help="User requirement file path for the benchmark (default: <dir_of_foambench_main.py>/user_requirement.txt)"
    )
    input_group.add_argument(
        '--case_path',
        type=str,
        default=None,
        help="Existing Foundation OpenFOAM v10 case directory or ZIP archive."
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
        help="Path to custom mesh file (e.g., .msh, .stl, .obj). If not provided, no custom mesh will be used."
    )
    parser.add_argument(
        '--overwrite_output',
        action='store_true',
        help="Explicitly allow replacing an existing non-empty output directory."
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help="Generate a PyVista visualization after a successful imported case run."
    )
    args = parser.parse_args()
    if args.case_subdir and not args.case_path:
        parser.error("--case_subdir requires --case_path.")
    if args.case_path and args.custom_mesh_path:
        parser.error("--custom_mesh_path is not available with --case_path.")
    if args.visualize and not args.case_path:
        parser.error("--visualize requires --case_path.")
    return args

def run_command(command, *, env=None):
    """
    Execute a command string using the current terminal's input/output,
    with the working directory set to the directory of the current file.
    
    Parameters:
        command: A command string or an argument sequence, e.g.
                 ``["python", "main.py", "--output_dir", "xxxx"]``.
        env: Optional environment overrides for the child workflow process.
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
            env=env,
        )
        print(f"Finished command: Return Code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(e.returncode)

def main():
    args = parse_args()
    print(args)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    child_env = os.environ.copy()
    if args.openfoam_path:
        openfoam_root = os.path.abspath(os.path.expanduser(args.openfoam_path))
        bashrc_path = os.path.join(openfoam_root, "etc", "bashrc")
        if not os.path.isfile(bashrc_path):
            raise ValueError(
                "--openfoam_path must point to a Foundation OpenFOAM installation "
                f"containing etc/bashrc: {openfoam_root}"
            )
        child_env["WM_PROJECT_DIR"] = openfoam_root
        print(f"Using OpenFOAM installation: {openfoam_root}")

    # Output creation and any overwrite decision are owned by the workflow's
    # ownership checks before the workflow gets a chance to validate it.

    # Build the workflow invocation as argument tokens, not shell text.
    main_cmd = [sys.executable, "src/main.py", "--output_dir", args.output]
    if args.case_path:
        main_cmd.extend(["--case_path", args.case_path])
        if args.case_subdir:
            main_cmd.extend(["--case_subdir", args.case_subdir])
        if args.visualize:
            main_cmd.append("--visualize")
    else:
        prompt_path = args.prompt_path or os.path.join(base_dir, "user_requirement.txt")
        main_cmd.extend(["--prompt_path", prompt_path])
    if args.custom_mesh_path:
        main_cmd.extend(["--custom_mesh_path", args.custom_mesh_path])
    if args.overwrite_output:
        main_cmd.append("--overwrite_output")
    
    print(f"Main workflow command: {shlex.join(main_cmd)}")
    
    print("Starting workflow...")
    if args.openfoam_path:
        run_command(main_cmd, env=child_env)
    else:
        # Preserve the original call shape for wrappers which replace
        # ``run_command`` and do not need an environment override.
        run_command(main_cmd)
    print("Workflow command finished.")

if __name__ == "__main__":
    # Examples (paths are resolved relative to the directory containing this file):
    #   python foambench_main.py
    #   python foambench_main.py --output output --prompt_path user_requirement.txt
    #   python foambench_main.py --output output --prompt_path user_requirement.txt --custom_mesh_path my_mesh.msh
    #   python foambench_main.py --output imported --case_path ./existing_case
    #   python foambench_main.py --output imported --case_path ./existing_case --visualize
    main()
