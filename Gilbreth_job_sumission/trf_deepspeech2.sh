#!/bin/bash
## Usage example: ./trf_deepspeech2.sh "-s -v -b 50 -i trf_300"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    # deepspeech2
    "-m deepspeech2 -l 0"
    "-m deepspeech2 -l 1"
    "-m deepspeech2 -l 2"
    "-m deepspeech2 -l 3"
    "-m deepspeech2 -l 4"
    "-m deepspeech2 -l 5"
    "-m deepspeech2 -l 6"
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
    esac

    # Submit the job with the current set of arguments
    echo trf_submit.sh $args $base_args$suff

    # Submit the job
    sbatch trf_submit.sh $args $base_args$suff
done