#!/bin/bash
#------------------------------------------------------------------------
#  run_predictor_local.sh - locally starts a predictor instance
#------------------------------------------------------------------------

# infer the project location
VM_DIR=$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ) )

if [ $# -eq 0 ]; then
    echo "Not enough arguments provided (multi or single as mode)"
    echo " e.g. ./run_local single -t cpu_usage -d single_model -f day weekday hour minute hist-1D VM_ID -d . data/train.csv "
    exit 1
fi
echo "Local run directory '$VM_DIR'..."

# inject into python path and run with existing args (for unix-like environments)
APP=$1
shift 1
PYTHONPATH="$VM_DIR:$PYTHONPATH" python $VM_DIR/bin/run_vm-predictor_reference_$APP.py $*
