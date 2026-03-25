import os
import hashlib
import json
from pathlib import Path

def generate_hash_for_dir(path):
    hashes = {}
    path = Path(path)
    # Recursively scan and hash all files
    for p in sorted(path.rglob('*')):
        if p.is_file():
            # Generate hash of file content
            with open(p, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            hashes[str(p.relative_to(path))] = file_hash
    return hashes

def main():
    # Directories to scan as per specification
    target_dirs = ['0', 'constant', 'system']
    nexus_data = {"case_verification": {}}
    
    for dir_name in target_dirs:
        if os.path.exists(dir_name):
            nexus_data["case_verification"][dir_name] = generate_hash_for_dir(dir_name)
    
    # Generate cumulative hash of all configuration
    full_content_str = json.dumps(nexus_data, sort_keys=True)
    nexus_data["cumulative_hash"] = hashlib.sha256(full_content_str.encode()).hexdigest()
    
    with open('verification_nexus.json', 'w') as f:
        json.dump(nexus_data, f, indent=4)
        
    print("Verification complete. Results saved to verification_nexus.json")

if __name__ == "__main__":
    main()
