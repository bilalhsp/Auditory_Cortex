#!/bin/bash
## Usage example: ./bootstrap_test.sh "-m whisper_base -l 2 -d ucdavis -b 50 -i exp_design_1"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define arbitrary lists of percent durations and trial IDs
percent_list=(20 40 60 80 100)
trial_ids=(1 2 3 4 5 6 7 8 9 10 11 12) 

# percent_list=(100)
# trial_ids=(12) 

# Clear previous args_sets
args_sets=()

# Build argument combinations
for percent in "${percent_list[@]}"; do
  for trial_id in "${trial_ids[@]}"; do
    args_sets+=("--test_bootstrap --n_test_trials $trial_id --percent_duration $percent")
  done
done

# Loop through each argument set and compute suffix
for idx in "${!args_sets[@]}"
do
    args=${args_sets[$idx]}

    num_trials=${#trial_ids[@]}
    trial_num_idx=$(( idx % num_trials ))
    duration_idx=$(( idx / num_trials ))

    trial_id=${trial_ids[$trial_num_idx]}
    duration=${percent_list[$duration_idx]}

    # Construct suffix
    suff="_tr_${trial_id}_dur_${duration}"

    # Echo or submit the job
    echo sbatch bootstrap_fit.sh $args $base_args$suff
    sbatch bootstrap_fit.sh $args $base_args$suff
done

