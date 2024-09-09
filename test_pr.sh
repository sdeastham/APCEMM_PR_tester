#!/bin/bash
#PBS -N APCEMM_PR_test
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=4:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

if [[ "z$pr_id" == "z" ]]; then
    echo "Pull request ID must be provided"
    exit 101
fi

#pr_id=$1

source apcemm.env

run_dir=$( readlink -f rundir_example )
mkdir test_${pr_id}
cd test_${pr_id}
mkdir Run

start_loc=$PWD

for codebase in base updated; do
    echo "Beginning work for $codebase"
    cd $start_loc
    echo " --> git operations"
    git clone https://github.com/mit-lae/APCEMM Code.APCEMM_$codebase &> log.git
    cd Code.APCEMM_$codebase
    if [[ "$codebase" == "updated" ]]; then
        git fetch origin pull/${pr_id}/head:pr_test >>../log.git 2>&1
        if [[ $? -ne 0 ]]; then
            echo " --> Fetch failed"
            continue
        fi
        git switch pr_test
        if [[ $? -ne 0 ]]; then
            echo " --> Switch failed"
            continue
        fi
    fi
    git submodule update --init --recursive >>../log.git 2>&1
    echo " --> code compilation"
    mkdir build
    cd build
    cmake ../Code.v05-00 &> log.cmake && cmake --build . &> log.build
    rc=$?
    if [[ $rc -ne 0 ]]; then
        " --> Build $codebase failed with return code $rc"
        continue
    fi
    APCEMM_binary=$( readlink -f APCEMM )
    echo " --> rundir creation"
    cd $start_loc/Run
    cp -a $run_dir $codebase
    cd $codebase
    ln -s $APCEMM_binary APCEMM
    ln -s ../../Code.APCEMM_$codebase/input_data
    echo " --> beginning run"
    ./APCEMM input.yaml &> log.run
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo " --> Running $codebase failed with return code $rc"
        continue
    fi
    echo " --> run successful"
    echo ""
done

cd $start_loc/..
./compare_runs.sh $pr_id
