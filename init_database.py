import os
import subprocess
import sys
import argparse
import shlex
import json
import shutil
from pathlib import Path
from scripts.verify_target_corpus import _is_lfs_pointer

ESI_V2006_SOURCE_URL = "https://dl.openfoam.com/source/v2006/OpenFOAM-v2006.tgz"
ESI_V2006_SOURCE_MD5 = "1226d48e74a4c78f12396cb586c331d8"
FOUNDATION_V10 = "foundation-v10"
ESI_V2006 = "esi-v2006"
TARGET_RUNTIME_VERSIONS = {
    FOUNDATION_V10: "10",
    ESI_V2006: "2006",
}
TARGET_SOURCES = {
    FOUNDATION_V10: {
        "release": "Foundation OpenFOAM v10",
        "reference_url": "https://openfoam.org/version/10/",
    },
    ESI_V2006: {
        "archive_url": ESI_V2006_SOURCE_URL,
        "archive_md5": ESI_V2006_SOURCE_MD5,
        "release": "ESI/OpenCFD OpenFOAM v2006",
    },
}
REQUIRED_RAW_CORPUS_FILES = (
    "openfoam_case_stats.json",
    "openfoam_commands.txt",
    "openfoam_command_help.txt",
    "openfoam_allrun_scripts.txt",
    "openfoam_tutorials_structure.txt",
    "openfoam_tutorials_details.txt",
)
FAISS_INDEX_NAMES = (
    "openfoam_command_help", "openfoam_allrun_scripts",
    "openfoam_tutorials_structure", "openfoam_tutorials_details",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize database for Foam-Agent project")
    parser.add_argument(
        '--openfoam_path',
        type=str,
        default=os.getenv("WM_PROJECT_DIR"),
        help="Path to OpenFOAM installation (WM_PROJECT_DIR)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force re-generate raw tutorial dumps + FAISS indices even if files exist"
    )
    parser.add_argument(
        '--raw_only',
        action='store_true',
        help="Generate only the raw tutorial corpus and target manifest; skip FAISS indices.",
    )
    parser.add_argument(
        '--openfoam_target',
        type=str,
        default=FOUNDATION_V10,
        choices=[FOUNDATION_V10, ESI_V2006],
        help=(
            "Build one target-scoped corpus. Defaults to foundation-v10; use "
            "esi-v2006 for native ESI/OpenCFD v2006 tutorials."
        ),
    )
    parser.add_argument(
        '--database_path',
        type=str,
        default='',
        help="Override the target corpus path. Defaults to database/<openfoam_target>.",
    )
    parser.add_argument(
        '--embedding_provider',
        type=str,
        default='',
        help="Optional embedding provider forwarded to the FAISS builders.",
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='',
        help="Optional embedding model forwarded to the FAISS builders.",
    )
    parser.add_argument(
        '--tutorials_path',
        type=str,
        default='',
        help=(
            "Optional tutorials root when it is outside the selected runtime tree."
        ),
    )
    return parser.parse_args()

def run_command(command_str, *, openfoam_root=None):
    """
    Execute a command string using the current terminal's input/output,
    with the working directory set to the directory of the current file.
    
    Parameters:
        command_str (str): The command to execute, e.g. "python main.py --output_dir xxxx" 
                           or "bash xxxxx.sh".
        openfoam_root: Source this installation before raw tutorial tools run.
    """
    # Split the command string into a list of arguments
    args = shlex.split(command_str)
    if openfoam_root is not None:
        args = [
            "bash", "--noprofile", "--norc", "-c",
            'foamagent_bashrc="$1"; shift; foamagent_command=("$@"); set --; '
            'source "$foamagent_bashrc" || exit $?; exec "${foamagent_command[@]}"',
            "foamagent-corpus", str(Path(openfoam_root) / "etc/bashrc"), *args,
        ]
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


def _embedding_index_dir(database_dir: Path, embedding_model: str) -> Path:
    """Return the FAISS model directory using the runtime's naming rule."""
    return database_dir / "faiss" / embedding_model.replace("/", "_").replace(":", "_")


def _corpus_file_is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0 and not _is_lfs_pointer(path)


def _raw_corpus_is_complete(raw_dir: Path) -> bool:
    return all(
        _corpus_file_is_complete(raw_dir / filename) for filename in REQUIRED_RAW_CORPUS_FILES
    )


def _faiss_index_is_complete(index_dir: Path) -> bool:
    return (
        _corpus_file_is_complete(index_dir / "index.faiss")
        and _corpus_file_is_complete(index_dir / "index.pkl")
    )


def _read_openfoam_version(openfoam_root: str, openfoam_target: str = FOUNDATION_V10) -> str:
    """Probe the selected installation using its native version contract."""
    root = Path(openfoam_root).expanduser().resolve()
    bashrc = root / "etc" / "bashrc"
    if not bashrc.is_file():
        raise ValueError(f"OpenFOAM bashrc not found at: {bashrc}")
    if openfoam_target == FOUNDATION_V10:
        environment = os.environ.copy()
        environment.pop("WM_PROJECT_VERSION", None)
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c",
             'source "$1" > /dev/null && [ "$WM_PROJECT_DIR" -ef "$2" ] '
             '&& printf "%s\\n" "$WM_PROJECT_VERSION"',
             "foamagent-version", str(bashrc), str(root)],
            check=True, capture_output=True, text=True, env=environment,
        )
        return result.stdout.strip()

    # ESI v2006 exposes its API directly. Its packaged bashrc can terminate a
    # noninteractive capture shell before a version print, so retain this API.
    foam_etc_file = root / "bin" / "foamEtcFile"
    if not foam_etc_file.is_file():
        raise ValueError(f"OpenFOAM foamEtcFile not found at: {foam_etc_file}")
    result = subprocess.run(
        [str(foam_etc_file), "-show-api"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

def main():
    args = parse_args()
    print(args)

    # Set environment variables
    WM_PROJECT_DIR = args.openfoam_path
    if not WM_PROJECT_DIR:
        raise ValueError("--openfoam_path (or WM_PROJECT_DIR) is required.")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"script_dir: {script_dir}")

    default_database = Path(script_dir) / "database"
    database_dir = (
        Path(args.database_path).expanduser().resolve()
        if args.database_path
        else default_database / args.openfoam_target
    )
    raw_dir = database_dir / "raw"
    database_arg = shlex.quote(str(database_dir))
    output_arg = shlex.quote(str(raw_dir))
    embedding_args = ""
    if args.embedding_provider:
        embedding_args += f" --embedding_provider={shlex.quote(args.embedding_provider)}"
    if args.embedding_model:
        embedding_args += f" --embedding_model={shlex.quote(args.embedding_model)}"

    selected_model = args.embedding_model or "Qwen/Qwen3-Embedding-0.6B"
    selected_index_dir = _embedding_index_dir(database_dir, selected_model)

    manifest_path = raw_dir / "foamagent_target.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict)
                or manifest.get("openfoam_target") != args.openfoam_target
                or str(manifest.get("wm_project_version", "")).lstrip("vV")
                != TARGET_RUNTIME_VERSIONS[args.openfoam_target]):
            raise ValueError(
                f"Corpus manifest {manifest_path} does not match {args.openfoam_target}. "
                "Use a separate directory for each native target."
            )
    elif (all((raw_dir / filename).is_file() for filename in REQUIRED_RAW_CORPUS_FILES)
          or (database_dir / "faiss").is_dir()):
        raise ValueError(
            f"Cannot establish the target of existing corpus assets without {manifest_path}. "
            "Use a separate directory to build a corpus with a verified native target."
        )

    existing_assets = [raw_dir / filename for filename in REQUIRED_RAW_CORPUS_FILES]
    if not args.raw_only:
        existing_assets.extend(selected_index_dir / index / filename
            for index in FAISS_INDEX_NAMES
            for filename in ("index.faiss", "index.pkl"))
    unavailable = [path for path in existing_assets if path.is_file() and not _corpus_file_is_complete(path)]
    if unavailable and not args.force:
        raise ValueError(
            "Existing corpus assets are empty or unhydrated Git LFS pointers: "
            + ", ".join(map(str, unavailable))
            + ". Run git lfs pull, or pass --force to explicitly rebuild this target/model."
        )

    runtime_version = _read_openfoam_version(str(WM_PROJECT_DIR), args.openfoam_target)
    normalized_version = runtime_version.lstrip("vV")
    expected_version = TARGET_RUNTIME_VERSIONS[args.openfoam_target]
    if normalized_version != expected_version:
        raise ValueError(
            f"--openfoam_target {args.openfoam_target} requires "
            f"{TARGET_SOURCES[args.openfoam_target]['release']}, but its runtime "
            f"reported WM_PROJECT_VERSION={runtime_version!r}."
        )

    # Keep parser input tied to the requested native runtime even if the
    # parent shell previously sourced another OpenFOAM distribution.
    runtime_root = Path(WM_PROJECT_DIR).expanduser().resolve()
    openfoam_arg = shlex.quote(str(runtime_root))
    os.environ["WM_PROJECT_DIR"] = str(runtime_root)
    tutorials_path = (
        Path(args.tutorials_path).expanduser().resolve()
        if args.tutorials_path
        else runtime_root / "tutorials"
    )
    if not tutorials_path.is_dir():
        raise ValueError(
            f"{TARGET_SOURCES[args.openfoam_target]['release']} tutorials directory is unavailable at "
            f"{tutorials_path}. Pass --tutorials_path <OpenFOAM-tutorials>."
        )
    os.environ["FOAM_TUTORIALS"] = str(tutorials_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    SCRIPTS = []
    
    # Preprocess the OpenFOAM tutorials
    regenerate_raw = args.force or not _raw_corpus_is_complete(raw_dir)
    if regenerate_raw:
        SCRIPTS.append((
            f"{shlex.quote(sys.executable)} database/script/tutorial_parser.py "
            f"--output_dir={output_arg} --wm_project_dir={openfoam_arg} "
            f"--tutorials_dir={shlex.quote(os.environ.get('FOAM_TUTORIALS', ''))}",
            runtime_root,
        ))

    # (Re)build FAISS indices
    # The embedding-model directory is part of the index identity.  Checking
    # only for *any* FAISS index made a second provider/model silently reuse a
    # missing target index instead of building it.
    python_executable = shlex.quote(sys.executable)
    if not args.raw_only:
        for index in FAISS_INDEX_NAMES:
            if regenerate_raw or not _faiss_index_is_complete(selected_index_dir / index):
                builder = index.removeprefix("openfoam_")
                SCRIPTS.append((f"{python_executable} database/script/faiss_{builder}.py --database_path={database_arg}{embedding_args}", None))

    if not SCRIPTS:
        print("All database files already exist. No initialization needed.")
        print("Tip: pass --force to rebuild.")
        return

    print("Starting database initialization...")
    for script, openfoam_root in SCRIPTS:
        if openfoam_root is not None:
            # Every embedding variant depends on the raw files this command
            # replaces. Invalidate before extraction, which can fail part-way.
            index_root = database_dir / "faiss"
            if index_root.is_symlink():
                raise ValueError(f"Cannot invalidate corpus indices through symbolic link: {index_root}")
            index_paths = [path for index in FAISS_INDEX_NAMES
                           for path in index_root.glob(f"*/{index}")]
            for path in index_paths:
                for component in (path.parent, path):
                    if component.is_symlink():
                        raise ValueError(f"Cannot invalidate corpus indices through symbolic link: {component}")
            for path in index_paths:
                shutil.rmtree(path)
        run_command(script, openfoam_root=openfoam_root)
        if openfoam_root is not None:
            if not _raw_corpus_is_complete(raw_dir):
                unavailable = [raw_dir / filename for filename in REQUIRED_RAW_CORPUS_FILES
                               if not _corpus_file_is_complete(raw_dir / filename)]
                raise ValueError(
                    "Raw corpus generation left missing, empty or unhydrated Git LFS assets: "
                    + ", ".join(map(str, unavailable))
                )
            # A manifest describes the raw corpus actually generated, not
            # merely the native installation selected for this invocation.
            manifest_path.write_text(
                json.dumps(
                    {
                        "openfoam_target": args.openfoam_target,
                        "wm_project_version": (
                            f"v{normalized_version}" if args.openfoam_target == ESI_V2006 else normalized_version
                        ),
                        "source": TARGET_SOURCES[args.openfoam_target],
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
    print("Database initialization completed successfully.")

if __name__ == "__main__":
    ## python init_database.py --openfoam_path $WM_PROJECT_DIR
    main()
