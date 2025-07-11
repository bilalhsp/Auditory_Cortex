#!/bin/bash
## Usage example: ./trf_speech2text.sh "-s -b 50 -i trf_300"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    # speech2text
    "-m speech2text -l 0"
    "-m speech2text -l 1"
    "-m speech2text -l 2"
    "-m speech2text -l 3"
    "-m speech2text -l 4"
    "-m speech2text -l 5"
    "-m speech2text -l 6"
    "-m speech2text -l 7"
    "-m speech2text -l 8"
    "-m speech2text -l 9"
    "-m speech2text -l 10"
    "-m speech2text -l 11"
    "-m speech2text -l 12"
    "-m speech2text -l 13"
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
    echo "sbatch trf_submit.sh $args $base_args$suff"

    # Submit the job
    sbatch trf_submit.sh $args $base_args$suff
done
