#!/bin/bash

# This script is used to submit bootstrap test jobs for different epochs.
# Example: 
#   To submit bootstrap test jobs for epochs 1 to 50,
#   we have to run following command for idx=[1, 2, ..., 50]:
#       ./bootstrap_test.sh "-m whisper_base -l 2 -d ucdavis -b 50 -i exp_design_idx"
# Usage: 
#    We can use this script to automate the submission of these jobs for a range of indices. 
#       ./bts_test_all_epochs.sh "-m whisper_base -l 2 -d ucdavis -b 50 -i exp_design_" 1 50

args=$1
START=$2
END=$3

if [[ -z "$START" || -z "$END" ]]; then
  echo "Usage: $0 \"<arg_prefix>\" <start_idx> <end_idx>"
  exit 1
fi

for (( i=START; i<=END; i++ )); do
  echo "Submitting job for idx = $i"
  ./bootstrap_test.sh "$args$i"
done
