#!/bin/bash
## Usage example: ./bootstrap_test.sh "-m whisper_base -l 2 -d ucdavis -b 50 -i itr"
# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shuffle or not> <bin_width> <results identifier>"
    exit 1
fi

base_args="$1"

# Define the sets of additional arguments for each submission
args_sets=(
    # whisper_base
    "--test_bootstrap --N_test_trials 1"
    "--test_bootstrap --N_test_trials 2"
    "--test_bootstrap --N_test_trials 3"
    # "--test_bootstrap --N_test_trials 4"
    # "--test_bootstrap --N_test_trials 5"
    # "--test_bootstrap --N_test_trials 6"
    # "--test_bootstrap --N_test_trials 7"
    # "--test_bootstrap --N_test_trials 8"
    # "--test_bootstrap --N_test_trials 9"
    # "--test_bootstrap --N_test_trials 10"
    # "--test_bootstrap --N_test_trials 11"
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
            suff="_1"
            ;;   
        1)
            suff="_2"
            ;;
        2)
            suff="_3"
            ;; 
    esac
        # 3)
        #     suff="_4"
        #     ;;
        # 4)
        #     suff="_5"
        #     ;;
        # 5)
        #     suff="_6"
        #     ;;
        # 6)
        #     suff="_7"
        #     ;;
        # 7)
        #     suff="_8"
        #     ;;
        # 8)
        #     suff="_9"
        #     ;;
        # 9)
        #     suff="_10"
        #     ;;
        # 10)
        #     suff="_11"
        #     ;;
    # esac

    # Submit the job with the current set of arguments
    echo bootstrap_fit.sh $args $base_args$suff

    # Submit the job
    sbatch bootstrap_fit.sh $args $base_args$suff
done
