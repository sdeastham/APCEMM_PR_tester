#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Need one argument - the pull request ID"
    exit 101
fi
pr_id=$1

qsub -v pr_id="$pr_id" test_pr.sh
