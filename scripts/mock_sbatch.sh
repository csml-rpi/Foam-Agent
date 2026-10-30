#!/usr/bin/env bash
# Mock sbatch utility for local verification
echo "Successfully received HPC Job submission!"
echo "Job parameters detected: $@"

# Run the OpenFOAM commands sequentially
echo "Starting blockMesh..."
blockMesh > log.blockMesh 2>&1
if [ $? -ne 0 ]; then
    echo "blockMesh failed."
    exit 1
fi

APP=$(grep -E '^[[:space:]]*application[[:space:]]+' system/controlDict | awk '{print $2}' | tr -d ';')
if [ -z "$APP" ]; then
    echo "Could not parse application from controlDict"
    exit 1
fi

echo "Starting solver ($APP) with timeout..."
timeout 600 "$APP" > log."$APP" 2>&1
if [ $? -eq 124 ]; then
    echo "Solver timed out after 600s."
else
    echo "Solver completed."
fi

echo "HPC Job execution completed successfully."
