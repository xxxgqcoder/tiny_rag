#!/usr/bin/env bash

set -e

echo "process started"

# project root dir
script=$(realpath "$0")
project_root_dir=$(dirname "$script")
echo "project root dir: $project_root_dir"


# start juypter for debug
sh start_jupyter.sh
echo 'jupyter server started'


# start python server
python -m rag.service


# hang for ever
tail -f /dev/null

wait
