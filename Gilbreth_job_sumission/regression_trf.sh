#!/bin/bash
## Usage example: ./regression_trf.sh "-m spect2vec -l 0 -b 50 -v -i mVocs_trf_lags300_l0"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    "--start 0 --end 5"
    "--start 5 --end 10"
    "--start 10 --end 15"
    "--start 15 --end 20"
    "--start 20 --end 25"
    "--start 25 --end 30"
    "--start 30 --end 34"
    "--start 34 --end 37"
    "--start 37"
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
    echo trf_submit.sh $args $base_args$suff


    sbatch trf_submit.sh $args $base_args$suff
done
