#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Need one argument - the pull request ID"
    exit 101
fi
pr_id=$1

job_id=$( qsub -v pr_id="$pr_id" test_pr.sh )
job_id=$( echo $job_id | cut -d'.' -f1 )

f_log=APCEMM_PR_test.o${job_id}
while [[ ! -f $f_log ]]; do
    echo "$( date ) -> waiting for job ${job_id} to begin logging in ${f_log}..."
    sleep 30
done
tail -f $f_log
