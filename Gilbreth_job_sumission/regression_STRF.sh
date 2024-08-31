#!/bin/bash
## Usage example: ./regression_STRF.sh "-b 50 -v -mel -i mVocs_wavlet_lags300"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    "-s 0 -e 5"
    "-s 5 -e 10"
    "-s 10 -e 15"
    "-s 15 -e 20"
    "-s 20 -e 25"
    "-s 25 -e 30"
    "-s 30 -e 34"
    "-s 34 -e 37"
    "-s 37"
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
            suff="_5_10"
            ;;
        2)
            suff="_10_15"
            ;;
        3)
            suff="_15_20"
            ;;
        4)
            suff="_20_25"
            ;;
        5)
            suff="_25_30"
            ;;
        6)
            suff="_30_34"
            ;;
        7)
            suff="_34_37"
            ;;
        8)
            suff="_37"
            ;;
    esac


    # Submit the job with the current set of arguments
    echo STRF_train.sh $args $base_args$suff


    sbatch STRF_train.sh $args $base_args$suff
done
