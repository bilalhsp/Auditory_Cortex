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
    # w2v2_audioset
    "-m w2v2_audioset -l 0"
    "-m w2v2_audioset -l 1"
    "-m w2v2_audioset -l 2"
    "-m w2v2_audioset -l 3"
    "-m w2v2_audioset -l 4"
    "-m w2v2_audioset -l 5"
    "-m w2v2_audioset -l 6"
    "-m w2v2_audioset -l 7"
    "-m w2v2_audioset -l 8"
    "-m w2v2_audioset -l 9"
    "-m w2v2_audioset -l 10"
    "-m w2v2_audioset -l 11"
    "-m w2v2_audioset -l 12"
    "-m w2v2_audioset -l 13"
    "-m w2v2_audioset -l 14"
    "-m w2v2_audioset -l 15"
    "-m w2v2_audioset -l 16"
    "-m w2v2_audioset -l 17"
    "-m w2v2_audioset -l 18"
    "-m w2v2_audioset -l 19"
    "-m w2v2_audioset -l 20"
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
