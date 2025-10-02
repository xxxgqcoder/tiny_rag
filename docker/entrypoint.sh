#!/usr/bin/env bash

set -e

echo "process started"

source .venv/bin/activate
python --version

# project root dir
script=$(realpath "$0")
project_root_dir=$(dirname "$script")
echo "project root dir: $project_root_dir"

# Add project root to PYTHONPATH
export PYTHONPATH="${project_root_dir}:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# start python server
nohup python -m rag.service 2>&1 &
echo "python server started"

# start file monitor
nohup python -m rag.document 2>&1 &
echo "file monitor started"


# hang for ever
tail -f /dev/null

wait
