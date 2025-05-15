#!/usr/bin/env bash

set -e

echo "process started"

sh start_jupyter.sh
echo 'jupyter server started'


# replace project root directory in MinerU config
sed 
echo "finish replace project root directory"

tail -f /dev/null

wait


