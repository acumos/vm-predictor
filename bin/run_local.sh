#!/bin/bash
#------------------------------------------------------------------------
#  run_predictor_local.sh - locally starts a predictor instance
#------------------------------------------------------------------------

# infer the project location
VM_DIR=$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ) )
echo "Local run directory '$VM_DIR'..."

# inject into python path and run with existing args (for unix-like environments)
PYTHONPATH="$VM_DIR:$PYTHONPATH" python $VM_DIR/bin/run_vm-predictor_reference.py $*
