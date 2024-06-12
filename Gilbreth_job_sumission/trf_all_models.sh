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
    # deepspeech2
    "-m deepspeech2 -l 0"
    "-m deepspeech2 -l 1"
    "-m deepspeech2 -l 2"
    "-m deepspeech2 -l 3"
    "-m deepspeech2 -l 4"
    "-m deepspeech2 -l 5"
    "-m deepspeech2 -l 6"
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
    # whisper_tiny
    "-m whisper_tiny -l 0"
    "-m whisper_tiny -l 1"
    "-m whisper_tiny -l 2"
    "-m whisper_tiny -l 3"
    "-m whisper_tiny -l 4"
    "-m whisper_tiny -l 5"
    # whisper_base
    "-m whisper_base -l 0"
    "-m whisper_base -l 1"
    "-m whisper_base -l 2"
    "-m whisper_base -l 3"
    "-m whisper_base -l 4"
    "-m whisper_base -l 5"
    "-m whisper_base -l 6"
    "-m whisper_base -l 7"
    # wav2vec2
    "-m wav2vec2 -l 0"
    "-m wav2vec2 -l 1"
    "-m wav2vec2 -l 2"
    "-m wav2vec2 -l 3"
    "-m wav2vec2 -l 4"
    "-m wav2vec2 -l 5"
    "-m wav2vec2 -l 6"
    "-m wav2vec2 -l 7"
    "-m wav2vec2 -l 8"
    "-m wav2vec2 -l 9"
    "-m wav2vec2 -l 10"
    "-m wav2vec2 -l 11"
    "-m wav2vec2 -l 12"
    "-m wav2vec2 -l 13"
    "-m wav2vec2 -l 14"
    "-m wav2vec2 -l 15"
    "-m wav2vec2 -l 16"
    "-m wav2vec2 -l 17"
    "-m wav2vec2 -l 18"
    "-m wav2vec2 -l 19"
    "-m wav2vec2 -l 20"
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
        # deepspeech2
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
        # speech2text
        7)
            suff="_l0"
            ;;        
        8)
            suff="_l1"
            ;;
        9)
            suff="_l2"
            ;; 
        10)
            suff="_l3"
            ;;
        11)
            suff="_l4"
            ;;
        12)
            suff="_l5"
            ;;
        13)
            suff="_l6"
            ;;
        14)
            suff="_l7"
            ;;        
        15)
            suff="_l8"
            ;;
        16)
            suff="_l9"
            ;; 
        17)
            suff="_l10"
            ;;
        18)
            suff="_l11"
            ;;
        19)
            suff="_l12"
            ;;
        20)
            suff="_l13"
            ;;
        # whisper_tiny
        21)
            suff="_l0"
            ;;        
        22)
            suff="_l1"
            ;;
        23)
            suff="_l2"
            ;; 
        24)
            suff="_l3"
            ;;
        25)
            suff="_l4"
            ;;
        26)
            suff="_l5"
            ;;
        # whisper_base
        27)
            suff="_l0"
            ;;        
        28)
            suff="_l1"
            ;;
        29)
            suff="_l2"
            ;; 
        30)
            suff="_l3"
            ;;
        31)
            suff="_l4"
            ;;
        32)
            suff="_l5"
            ;;
        33)
            suff="_l6"
            ;;
        34)
            suff="_l7"
            ;;
        #wav2vec2
        35)
            suff="_l0"
            ;;        
        36)
            suff="_l1"
            ;;
        37)
            suff="_l2"
            ;; 
        38)
            suff="_l3"
            ;;
        39)
            suff="_l4"
            ;;
        40)
            suff="_l5"
            ;;
        41)
            suff="_l6"
            ;;
        42)
            suff="_l7"
            ;;
        43)
            suff="_l8"
            ;;        
        44)
            suff="_l9"
            ;;
        45)
            suff="_l10"
            ;; 
        46)
            suff="_l11"
            ;;
        47)
            suff="_l12"
            ;;
        48)
            suff="_l13"
            ;;
        49)
            suff="_l14"
            ;;
        50)
            suff="_l15"
            ;; 
        51)
            suff="_l16"
            ;;
        52)
            suff="_l17"
            ;;
        53)
            suff="_l18"
            ;;
        54)
            suff="_l19"
            ;;
        55)
            suff="_l20"
            ;;
        # wav2letter_modified
        56)
            suff="_l0"
            ;;        
        57)
            suff="_l1"
            ;;
        58)
            suff="_l2"
            ;; 
        59)
            suff="_l3"
            ;;
        60)
            suff="_l4"
            ;;
        61)
            suff="_l5"
            ;;
        62)
            suff="_l6"
            ;;
        63)
            suff="_l7"
            ;;
        64)
            suff="_l8"
            ;;        
        65)
            suff="_l9"
            ;;
        66)
            suff="_l10"
            ;; 
        67)
            suff="_l11"
            ;;
        68)
            suff="_l12"
            ;;
        69)
            suff="_l13"
            ;;
        #w2v2_audioset
        70)
            suff="_l0"
            ;;        
        71)
            suff="_l1"
            ;;
        72)
            suff="_l2"
            ;; 
        73)
            suff="_l3"
            ;;
        74)
            suff="_l4"
            ;;
        75)
            suff="_l5"
            ;;
        76)
            suff="_l6"
            ;;
        77)
            suff="_l7"
            ;;
        78)
            suff="_l8"
            ;;        
        79)
            suff="_l9"
            ;;
        80)
            suff="_l10"
            ;; 
        81)
            suff="_l11"
            ;;
        82)
            suff="_l12"
            ;;
        83)
            suff="_l13"
            ;;
        84)
            suff="_l14"
            ;;
        85)
            suff="_l15"
            ;; 
        86)
            suff="_l16"
            ;;
        87)
            suff="_l17"
            ;;
        88)
            suff="_l18"
            ;;
        89)
            suff="_l19"
            ;;
        90)
            suff="_l20"
            ;;
    esac

    # Submit the job with the current set of arguments
    echo trf_submit.sh $args $base_args$suff

    # Submit the job
    sbatch trf_submit.sh $args $base_args$suff
done
