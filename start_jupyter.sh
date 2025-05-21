#!/usr/bin/env bash
set -e

jupyter_log_file='.logs/jupyter.logs'
port_num=4000

ps ax | grep -E "jupyter-lab --port=${port_num}" | grep -v "grep" | awk '{print $1}' | xargs -I {} kill -9 {}

rm -r ${jupyter_log_file} || true

mkdir -p $(dirname ${jupyter_log_file})

touch ${jupyter_log_file}

nohup jupyter-lab --port=${port_num} \
    --allow-root \
    --ip=0.0.0.0 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    > ${jupyter_log_file} 2>&1 &

echo 'jupyter-lab is running'