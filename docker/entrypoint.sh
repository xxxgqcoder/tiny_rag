#!/usr/bin/env bash

set -e

echo "process started"

sh start_jupyter.sh
echo 'jupyter server started'

tail -f /dev/null

wait