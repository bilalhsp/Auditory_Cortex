#!/bin/bash
## Usage example: ./bootstrap_whisper_base.sh "-b 50 -i 1"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    # whisper_base
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 1"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 2"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 3"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 4"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 5"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 6"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 7"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 8"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 9"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 10"
    "-m whisper_base -l 2 --test_bootstrap --N_test_trials 11"
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
            suff="_N1"
            ;;   
        1)
            suff="_N2"
            ;;
        2)
            suff="_N3"
            ;; 
        3)
            suff="_N4"
            ;;
        4)
            suff="_N5"
            ;;
        5)
            suff="_N6"
            ;;
        6)
            suff="_N7"
            ;;
        7)
            suff="_N8"
            ;;
        8)
            suff="_N9"
            ;;
        9)
            suff="_N10"
            ;;
        10)
            suff="_N11"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo bootstrap_submit.sh $args $base_args$suff

    # Submit the job
    sbatch bootstrap_submit.sh $args $base_args$suff
done
