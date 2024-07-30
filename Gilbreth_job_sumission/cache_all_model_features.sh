#!/bin/bash
## Usage example: 
# For loading features of shuffled networks
# ./cache_all_model_features.sh "-s"
# For loading features of pretrained networks
# ./cache_all_model_features.sh
# For loading features for mVocs
# ./cache_all_model_features.sh "-v"

# # Define the sets of additional arguments for each submission
args_sets=(
    "-i 0"
    "-i 1"
    "-i 2"
    "-i 3"
    "-i 4"
    "-i 5"
    "-i 7"
)
# # Replace the placeholder with the actual base identifier

# # Loop through each set of additional arguments and submit the job
for idx in "${!args_sets[@]}"
do
    args=${args_sets[$idx]}
    # $base_args+"_extended"
    
    # Submit the job with the current set of arguments
    echo cache_features.sh $args $1

    # sbatch regression_submit.sh $args $base_args
    sbatch cache_features.sh $args $1
done

# echo "Commands are only displayed, no job submitted."
