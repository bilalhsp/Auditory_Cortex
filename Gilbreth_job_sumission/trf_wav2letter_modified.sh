#!/bin/bash
## Usage example: ./trf_all_models.sh "-s -b 20 -i trained_all_bins"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    # wav2letter_modified
    "-m wav2letter_modified -l 0"
    "-m wav2letter_modified -l 1"
    "-m wav2letter_modified -l 2"
    "-m wav2letter_modified -l 3"
    "-m wav2letter_modified -l 4"
    "-m wav2letter_modified -l 5"
    "-m wav2letter_modified -l 6"
    "-m wav2letter_modified -l 7"
    "-m wav2letter_modified -l 8"
    "-m wav2letter_modified -l 9"
    "-m wav2letter_modified -l 10"
    "-m wav2letter_modified -l 11"
    "-m wav2letter_modified -l 12"
    "-m wav2letter_modified -l 13"
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
            suff="_l0"
            ;;        
        1)
            suff="_l1"
            ;;
        2)
            suff="_l2"
            ;; 
        3)
            suff="_l3"
            ;;
        4)
            suff="_l4"
            ;;
        5)
            suff="_l5"
            ;;
        6)
            suff="_l6"
            ;;
        7)
            suff="_l7"
            ;;
        8)
            suff="_l8"
            ;;        
        9)
            suff="_l9"
            ;;
        10)
            suff="_l10"
            ;; 
        11)
            suff="_l11"
            ;;
        12)
            suff="_l12"
            ;;
        13)
            suff="_l13"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo trf_submit.sh $args $base_args$suff

    # Submit the job
    sbatch trf_submit.sh $args $base_args$suff
done
