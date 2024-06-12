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
    "-m whisper_tiny"
    "-m whisper_base"
    "-m wav2letter_modified -l 0 1 2 3 4 5 6 7 8 9 10 11"
    "-m speech2text -l 2 3 4 5 6 7 8 9 10 11 12 13"
    "-m deepspeech2 -l 2 3 4 5 6"
    "-m wav2vec2 -l 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    "-m wav2vec2 -l 0 1 2 3 4 5 6"    
    "-m speech2text -l 0"
    "-m speech2text -l 1"
    "-m deepspeech2 -l 0"
    "-m deepspeech2 -l 1"
    "-m wav2letter_modified -l 12 13"
    "-m w2v2_audioset -l 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    "-m w2v2_audioset -l 0 1 2 3 4 5 6"
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
        6)
            suff="_features"
            ;;        
        7)
            suff="_l0"
            ;;
        8)
            suff="_l1"
            ;; 
        9)
            suff="_l0"
            ;;
        10)
            suff="_l1"
            ;;
        11)
            suff="_lasttwo"
            ;;
        13)
            suff="_features"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo regression_submit.sh $args $base_args$suff

    # sbatch regression_submit.sh $args $base_args
    sbatch regression_submit.sh $args $base_args$suff
done
