#!/bin/bash
## Usage example: ./regression_all_models.sh "-s -b 20 -i trained_all_bins"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    "-m CLAP -l 0 1"
    "-m CLAP -l 2 3"
    "-m CLAP -l 4 5"
    "-m CLAP -l 6 7"
) 
# Replace the placeholder with the actual base identifier

# Loop through each set of additional arguments and submit the job
for idx in "${!args_sets[@]}"
do
    args=${args_sets[$idx]}
    # $base_args+"_extended"
    suff=""
    # Append different suffixes based on the index of the current set of arguments
    case $idx in
        1)
            suff="_layers2_3"
            ;;
        2)
            suff="_layers4_5"
            ;;
        3)
            suff="_layers6_7"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo regression_submit.sh $args $base_args$suff

    # sbatch regression_submit.sh $args $base_args
    sbatch regression_submit.sh $args $base_args$suff
done
