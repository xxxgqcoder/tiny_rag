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


# replace project root directory
sed -i -e "s%<project_root_dir>%$project_root_dir%g" assets/MinerU/magic-pdf.json

sed -i -e "s%<project_root_dir>%$project_root_dir%g" assets/bge-m3/bge-m3.json
echo "finish replace project root directory"

# start python server
# python start_server.py

# hang for ever
tail -f /dev/null

wait
