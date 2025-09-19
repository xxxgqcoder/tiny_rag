#!/usr/bin/env bash

set -e
echo $(pwd)

file_path="/Users/xcoder/obsidian/Profession/PDF/FTRL algorithm.pdf"
final_md_file_save_dir="/Users/xcoder/obsidian/Profession/Processed PDF"

python -m rag.pipeline.process_pdf \
    --file_path="${file_path}" \
    --final_md_file_save_dir="${final_md_file_save_dir}"