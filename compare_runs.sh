#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Provide PR ID"
    exit 102
fi
pr_id=$1

source ~/.bashrc
conda activate gcpy
python3 compare_runs.py $pr_id
