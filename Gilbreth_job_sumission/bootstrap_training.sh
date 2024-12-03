#!/bin/bash
## Usage example: ./bootstrap_training.sh "-m whisper_base -l 2 -d ucdavis -b 50 -i itr"

# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    "-N 10"
    "-N 20"
    "-N 30"
    "-N 40"
    "-N 50"
    "-N 60"
    "-N 70"
    "-N 80"
    "-N 90"
    "-N 100"
)
# # Replace the placeholder with the actual base identifier

# Loop through each set of additional arguments and submit the job
for idx in "${!args_sets[@]}"
do
    args=${args_sets[$idx]}
    # $base_args+"_extended"
    suff=""
    # Append different suffixes based on the index of the current set of arguments
    case $idx in 
        0)
            suff="_10"
            ;;   
        1)
            suff="_20"
            ;;
        2)
            suff="_30"
            ;; 
        3)
            suff="_40"
            ;;
        4)
            suff="_50"
            ;;
        5)
            suff="_60"
            ;;
        6)
            suff="_70"
            ;;
        7)
            suff="_80"
            ;;
        8)
            suff="_90"
            ;;
        9)
            suff="_100"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo bootstrap_fit.sh $args $base_args$suff

    # Submit the job
    sbatch bootstrap_fit.sh $args $base_args$suff
done
