#!/usr/bin/env bash

set -e

echo "process started"

script=$(realpath "$0")
project_root_dir=$(dirname "$script")
echo "project root dir: $project_root_dir"

sh start_jupyter.sh
echo 'jupyter server started'

# replace project root directory in MinerU config
sed -i -e "s%<project_root_dir>%$project_root_dir%g" assets/MinerU/magic-pdf.json
echo "finish replace project root directory"

# hang for ever
tail -f /dev/null

wait


