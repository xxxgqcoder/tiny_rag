#!/bin/bash
set -e 

cd $(dirname "$0")


# format python
find . -type f -name '*py' \( -not -path "*/__pycache__/*" -and -not -path "*/.git/*" -and -not -path "*/ipynb_checkpoints/*" \) -prune \
    -exec yapf -i {} +
