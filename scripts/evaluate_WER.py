

import time
import argparse

from auditory_cortex.analyses.evaluate_pretrained_performance import compute_WER
from auditory_cortex.io_utils.io import read_WER, write_WER



def compute_and_save_WER(args):

    model_name = args.model_name
    benchmark = args.benchmark
    batch_size = 8
    wer = compute_WER(
        model_name=model_name,
        benchmark=benchmark,
        batch_size=batch_size
    )

    write_WER(model_name, benchmark, wer)

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save WER for pretrained models ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # add arguments to read from command line
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',#  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec2', 'speech2text', 'deepspeech2',
                'whisper_tiny', 'whisper_small', 'whisper_base', 'whisper_medium'],
        # default='wav2letter_modified', 
        help='model to be evaluated .'
    )
    parser.add_argument(
        '-b','--benchmark', dest='benchmark', type= str, action='store',
        choices=['librispeech-test-clean', 'librispeech-test-other','tedlium', 'common-voice',
                 'voxpopuli'],
        help="Specify the benchmark to compute WER on."
    )
    return parser



# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_WER(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")