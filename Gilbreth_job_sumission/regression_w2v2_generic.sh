#!/bin/bash
## Usage example: ./regression_w2v2_generic.sh "-s -b 20 -i trained_all_bins"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    "-m w2v2_generic -l 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    # "-m w2v2_generic -l 0 1 2 3 4 5 6"
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
            suff="_features"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo regression_submit.sh $args $base_args$suff

    # sbatch regression_submit.sh $args $base_args
    sbatch regression_submit.sh $args $base_args$suff
done
