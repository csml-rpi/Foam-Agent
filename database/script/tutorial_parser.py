import os
import subprocess
import argparse
import concurrent.futures
from pathlib import Path
import re
import json

def read_files_into_dict(base_path, stats=None):
    """Read OpenFOAM tutorial case files.

    Important: OpenFOAM cases often have *nested* region subfolders, e.g.
      - 0/air, 0/porous
      - constant/air, constant/porous
      - system/air, system/porous
    and they may contain duplicate filenames across regions (e.g. T, p).
    """
    if stats is None:
        stats = {
            "files_total_scanned": 0,
            "files_skipped_encoding": 0,
            "files_skipped_large": 0,
            "files_read_success": 0,
            "allrun_read_success": 0,
            "allrun_read_fail": 0
        }

    # Each entry: {"folder_name": <relative folder>, "file_name": <basename>, "content": <text>}
    entries = []

    # Read 'Allrun' file
    allrun_path = os.path.join(base_path, "Allrun")
    allrun_content = "None"
    
    # Check if "Allrun" exists and attempt to read it
    if os.path.isfile(allrun_path):
        stats["files_total_scanned"] += 1  # We are scanning the Allrun file
        
        try:
            with open(allrun_path, "r") as file_handle:
                allrun_content = file_handle.read()
            stats["allrun_read_success"] += 1
        except UnicodeDecodeError:
            print(f"Skipping file due to encoding error: {allrun_path}")
            stats["files_skipped_encoding"] += 1
            stats["allrun_read_fail"] += 1
        except Exception as e:
            print(f"Error reading file {allrun_path}: {e}")
            stats["allrun_read_fail"] += 1

    # Traverse the case directory recursively.
    # Keep folder_name relative to the case root (base_path).
    for root, _, files in os.walk(base_path):
        for file in files:
            # Skip the Allrun file (already handled above)
            if root == base_path and file == "Allrun":
                continue

            file_path = os.path.join(root, file)
            rel_folder = os.path.relpath(root, base_path)

            # Avoid sucking in huge decomposed meshes or processor dirs if present
            # (tutorials shouldn't have them, but generated artifacts might).
            if rel_folder.startswith("processor") or rel_folder.startswith("postProcessing"):
                continue

            stats["files_total_scanned"] += 1

            try:
                with open(file_path, "r") as file_handle:
                    content = file_handle.read()
                entries.append({
                    "folder_name": rel_folder,
                    "file_name": file,
                    "content": content
                })
                stats["files_read_success"] += 1
            except UnicodeDecodeError:
                print(f"Skipping file due to encoding error: {file_path}")
                stats["files_skipped_encoding"] += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return allrun_content, entries, stats


def find_cases(root_dir):
    """
    Traverse the directory tree under 'root_dir' and look for cases containing a 'system' folder.
    For each case found, extract metadata such as case name, solver, category, and domain.
    
    Additionally, collect statistics in a "funnel-like" manner to see how many directories 
    and files are processed, skipped due to encoding issues, skipped due to large size, etc.
    """
    cases = []
    
    # Initialize statistics dictionary
    stats = {
        "directories_scanned": 0,
        "directories_with_system": 0,
        "files_total_scanned": 0,
        "files_skipped_encoding": 0,
        "files_skipped_large": 0,
        "files_read_success": 0,
        "allrun_read_success": 0,
        "allrun_read_fail": 0
    }

    # Get FOAM_TUTORIALS from environment or fallback
    FOAM_TUTORIALS = os.environ.get("FOAM_TUTORIALS", "/home/somasn/Documents/LLM/OpenFOAM-10/tutorials")
    blockmesh_resource_dir = os.path.join(FOAM_TUTORIALS, "resources", "blockMesh")

    for root, dirs, _ in os.walk(root_dir):
        stats["directories_scanned"] += 1  # Scanning this directory

        # Check if the current directory contains a 'system' folder
        if "system" in dirs:
            stats["directories_with_system"] += 1

            # Read files in the current directory (root)
            allrun_content, entries, file_stats = read_files_into_dict(root, stats={
                "files_total_scanned": 0,
                "files_skipped_encoding": 0,
                "files_skipped_large": 0,
                "files_read_success": 0,
                "allrun_read_success": 0,
                "allrun_read_fail": 0
            })
            
            # Merge file_stats into the global stats
            stats["files_total_scanned"] += file_stats["files_total_scanned"]
            stats["files_skipped_encoding"] += file_stats["files_skipped_encoding"]
            stats["files_skipped_large"] += file_stats["files_skipped_large"]
            stats["files_read_success"] += file_stats["files_read_success"]
            stats["allrun_read_success"] += file_stats["allrun_read_success"]
            stats["allrun_read_fail"] += file_stats["allrun_read_fail"]

            # The case name is the name of the current directory
            case_name = os.path.basename(root)
            
            # Initialize solver, category, and domain
            solver, category, domain = None, None, None
            
            # Move up to the parent directory and search up to 3 levels
            current_path = os.path.dirname(root)
            found_foam = False

            for level in range(3):
                # Stop if the path is empty or if we have reached the root_dir
                if (not current_path) or (os.path.basename(current_path) == os.path.basename(root_dir)):
                    break
                
                dir_name = os.path.basename(current_path)
                
                # If the directory name ends with 'Foam', treat it as the solver
                if dir_name.endswith("Foam"):
                    solver = dir_name
                    # The parent of the solver directory is considered the domain
                    domain = os.path.basename(os.path.dirname(current_path))
                    found_foam = True
                    break
                elif level == 0:
                    category = dir_name
                
                # Move one level up
                current_path = os.path.dirname(current_path)
            
            # If no solver directory ending with 'Foam' was found, use the relative path logic
            if not found_foam:
                category = None  # Reset category in case it was partially set above
                relative_path = os.path.relpath(root, root_dir)
                path_components = relative_path.split(os.sep)
                
                # If the relative path has exactly 3 components: domain/solver/caseName
                if len(path_components) == 3:
                    domain, solver = path_components[0], path_components[1]
                # If the relative path has exactly 4 components: domain/solver/category/caseName
                elif len(path_components) == 4:
                    domain, solver, category = path_components[0], path_components[1], path_components[2]

            # --- NEW LOGIC: Check for missing blockMeshDict and copy if referenced in Allrun ---
            # (entries can include nested folders; entry_names is for quick checks)
            entry_names = [e.get("file_name") for e in entries]

            system_dir = os.path.join(root, "system")
            blockmeshdict_path = os.path.join(system_dir, "blockMeshDict")
            if not os.path.isfile(blockmeshdict_path):
                # Only try if Allrun exists and was read
                if allrun_content != "None":
                    # Look for blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/<name>
                    pattern = r"blockMesh\s+-dict\s+\$FOAM_TUTORIALS/resources/blockMesh/([\w\d_]+)"
                    match = re.search(pattern, allrun_content)
                    if match:
                        referenced_file = match.group(1)
                        src_blockmeshdict = os.path.join(blockmesh_resource_dir, referenced_file)
                        if os.path.isfile(src_blockmeshdict):
                            # Copy to system/blockMeshDict
                            try:
                                with open(src_blockmeshdict, "r") as src_f:
                                    blockmesh_content = src_f.read()
                                # Save to the case's system dir
                                os.makedirs(system_dir, exist_ok=True)
                                with open(blockmeshdict_path, "w") as dst_f:
                                    dst_f.write(blockmesh_content)
                                # Add to in-memory structures for output
                                entries.append({
                                    "folder_name": "system",
                                    "file_name": "blockMeshDict",
                                    "content": blockmesh_content
                                })
                                entry_names.append("blockMeshDict")
                                print(f"[INFO] Copied {src_blockmeshdict} to {blockmeshdict_path} for case {case_name}")
                            except Exception as e:
                                print(f"[WARNING] Failed to copy {src_blockmeshdict} to {blockmeshdict_path}: {e}")
                        else:
                            print(f"[WARNING] Referenced blockMeshDict {src_blockmeshdict} not found for case {case_name}")
                    else:
                        print(f"[INFO] No blockMesh -dict reference found in Allrun for case {case_name}")
                else:
                    print(f"[INFO] No Allrun file to check for blockMeshDict reference in case {case_name}")
            # --- END NEW LOGIC ---

            # Append the extracted metadata to the 'cases' list
            cases.append({
                "case_name": case_name,
                "solver": solver,
                "category": category,
                "domain": domain,
                "entries": entries,
                "allrun": allrun_content
            })
    
    return cases, stats



def save_cases_to_file(cases, output_dir):
    """
    Saves case details, summary, or Allrun content to a file.
    """
    
    allrun_filepath = f"{output_dir}/openfoam_allrun_scripts.txt"
    tutorials_summary_filepath = f"{output_dir}/openfoam_tutorials_structure.txt"
    tutorial_filepath = f"{output_dir}/openfoam_tutorials_details.txt"
    case_stats_filepath = f"{output_dir}/openfoam_case_stats.json"
    
    allrun_text = ''
    tutorials_summary_text = ''
    tutorials_text = ''
    
    case_stats = {
        'case_domain': set(),
        'case_category': set(),
        'case_solver': set()
    }
    
    for case in cases:
        case_name, case_domain, case_category, case_solver = (
            case["case_name"], case["domain"], case["category"], case["solver"]
        )
        
        if case_domain:
            case_stats['case_domain'].add(case_domain)
        if case_category:
            case_stats['case_category'].add(case_category)
        if case_solver:
            case_stats['case_solver'].add(case_solver)
        
        # Save the case index
        case_index_text = "<index>\n"
        case_index_text += f"case name: {case_name}\n"
        case_index_text += f"case domain: {case_domain}\n"
        case_index_text += f"case category: {case_category}\n"
        case_index_text += f"case solver: {case_solver}\n"
        case_index_text += "</index>\n\n"
        
        # Save the directory structure (folder -> list of filenames)
        folder_file_dict = {}
        for ent in case.get("entries", []):
            folder_name = ent.get("folder_name", "")
            file_name = ent.get("file_name", "")
            if not folder_name or not file_name:
                continue
            folder_file_dict.setdefault(folder_name, []).append(file_name)

        # Deterministic ordering for stable diffs
        for k in list(folder_file_dict.keys()):
            folder_file_dict[k] = sorted(set(folder_file_dict[k]))
        
        dir_structure_text = "<directory_structure>\n"
        for folder_name, file_names in folder_file_dict.items():
            dir_structure_text += f"<dir>directory name: {folder_name}. "
            dir_structure_text += f"File names in this directory: [{', '.join(file_names)}]</dir>\n"
        dir_structure_text += "</directory_structure>\n\n"
        
        
        if case["allrun"] != "None":
            # Save the Allrun content
            allrun_text += f'''
<case_begin>
{case_index_text}
{dir_structure_text}
<allrun_script>
{case["allrun"]}
</allrun_script>
</case_end>\n\n\n
'''

        # Save the tutorials summary
        tutorials_summary_text += f"<case_begin>\n{case_index_text}\n{dir_structure_text}\n</case_end>\n\n"

        # Save the detailed tutorials
        tutorials_text += f"<case_begin>\n{case_index_text}\n{dir_structure_text}\n<tutorials>\n"
        
        for folder_name, file_names in folder_file_dict.items():
            tutorials_text += f"<directory_begin>directory name: {folder_name}\n"
            for file_name in file_names:
                tutorials_text += f"<file_begin>file name: {file_name}\n"

                # Find the matching entry content (first match is fine; content should be identical per folder/file)
                content = ""
                for ent in case.get("entries", []):
                    if ent.get("folder_name") == folder_name and ent.get("file_name") == file_name:
                        content = ent.get("content", "")
                        break

                # Delete comments, such as license information, from the file contents
                cleaned_text = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                cleaned_text = re.sub(r'//.*', '', cleaned_text)

                tutorials_text += f"<file_content>{cleaned_text}</file_content>\n"
                tutorials_text += f"</file_end>\n\n"
            
            tutorials_text += f"</directory_end>\n\n"            

        tutorials_text += "</tutorials>\n</case_end>\n\n\n"

    with open(allrun_filepath, "w", encoding="utf-8") as file:
        file.write(allrun_text)
    
    with open(tutorials_summary_filepath, "w", encoding="utf-8") as file:
        file.write(tutorials_summary_text)
            
    with open(tutorial_filepath, "w", encoding="utf-8") as file:
        file.write(tutorials_text)
    
    case_stats['case_category'].add("None")
    case_stats['case_category'] = list(case_stats['case_category'])
    case_stats['case_domain'] = list(case_stats['case_domain'])
    case_stats['case_solver'] = list(case_stats['case_solver'])
    
    with open(case_stats_filepath, "w", encoding="utf-8") as file:
        json.dump(case_stats, file, ensure_ascii=False, indent=4)
            

def get_commands_from_directory(directory_path):
    """Retrieves all command file names from a specified directory using os.scandir."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    return [entry.name for entry in os.scandir(directory_path) if entry.is_file()]

def get_command_help(command, directory_path):
    """Retrieves the help message for a given command."""
    try:
        result = subprocess.run(
            f"{os.path.join(directory_path, command)} -help", shell=True, capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)

def fetch_command_helps(commands, directory_path):
    """Fetch help messages in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return dict(zip(commands, executor.map(lambda cmd: get_command_help(cmd, directory_path), commands)))

if __name__ == "__main__":
    # python ./database/script/tutorial_parser.py --output_dir=./database/foundation-v10/raw --wm_project_dir=$WM_PROJECT_DIR
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_project_dir", required=True, help="Path to WM_PROJECT_DIR")
    parser.add_argument("--output_dir", default='./database/foundation-v10/raw', help="Directory to save output files")
    parser.add_argument(
        "--tutorials_dir",
        default="",
        help="Optional tutorials root; defaults to <wm_project_dir>/tutorials.",
    )
    args = parser.parse_args()
    
    print(args)

    tutorial_path = args.tutorials_dir or os.path.join(args.wm_project_dir, "tutorials")
    if not os.path.isdir(tutorial_path):
        raise FileNotFoundError(f"OpenFOAM tutorials directory does not exist: {tutorial_path}")
    cases_info, case_stats = find_cases(tutorial_path)
    print(f"Statistics: {case_stats}")
    print(f"Found {len(cases_info)} cases in {tutorial_path}")
    

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cases_to_file(cases_info, output_dir)

    commands_path = Path(args.wm_project_dir) / "platforms/linux64GccDPInt32Opt/bin"
    commands = get_commands_from_directory(commands_path)
    command_help_data = fetch_command_helps(commands, commands_path)

    with open(output_dir / "openfoam_commands.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(commands) + "\n")

    with open(output_dir / "openfoam_command_help.txt", "w", encoding="utf-8") as f:
        for cmd, help_text in command_help_data.items():
            f.write(f"<command_begin><command>{cmd}</command><help_text>{help_text}</help_text></command_end>\n\n")
