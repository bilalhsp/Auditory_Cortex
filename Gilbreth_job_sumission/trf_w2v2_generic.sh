#!/bin/bash
## Usage example: ./trf_deepspeech2.sh "-s -b 50 -i trf_300"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    # w2v2_generic
    "-m w2v2_generic -l 0"
    "-m w2v2_generic -l 1"
    "-m w2v2_generic -l 2"
    "-m w2v2_generic -l 3"
    "-m w2v2_generic -l 4"
    "-m w2v2_generic -l 5"
    "-m w2v2_generic -l 6"
    "-m w2v2_generic -l 7"
    "-m w2v2_generic -l 8"
    "-m w2v2_generic -l 9"
    "-m w2v2_generic -l 10"
    "-m w2v2_generic -l 11"
    "-m w2v2_generic -l 12"
    "-m w2v2_generic -l 13"
    "-m w2v2_generic -l 14"
    "-m w2v2_generic -l 15"
    "-m w2v2_generic -l 16"
    "-m w2v2_generic -l 17"
    "-m w2v2_generic -l 18"
    "-m w2v2_generic -l 19"
    "-m w2v2_generic -l 20"
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
        14)
            suff="_l14"
            ;;
        15)
            suff="_l15"
            ;; 
        16)
            suff="_l16"
            ;;
        17)
            suff="_l17"
            ;;
        18)
            suff="_l18"
            ;;
        19)
            suff="_l19"
            ;;
        20)
            suff="_l20"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo trf_submit.sh $args $base_args$suff

    # Submit the job
    sbatch trf_submit.sh $args $base_args$suff
done
