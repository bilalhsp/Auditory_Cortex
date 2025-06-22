import time
import argparse

import auditory_cortex.deprecated.models as models
from auditory_cortex.io_utils.io import write_context_dependent_normalizer


# ------------------  Normalizer computing function ----------------------#

def compute_and_save_context_dependent_normalizers(args):

    # create an object for the normalizer
    n_iterations = 10000
    layer_IDs = args.layer_IDs
    bin_widths = args.bin_widths
    model_name = args.model_name

    obj = models.Regression(
                model_name=model_name
            )

    if layer_IDs is None:
        layer_IDs = obj.get_layer_IDs() 
    
    for bin_width in bin_widths:
        print(f"Computing for bin_width: {bin_width}...")
        layer_wise_normalizers = obj.compute_context_dependent_normalizer_variance(
            layer_IDs, bin_width=bin_width, n_iterations=n_iterations
        )

        # writing results to depot..
        write_context_dependent_normalizer(obj.model_name, layer_wise_normalizers, bin_width)

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save the context dependent '+
        'variance of the normalizer using ANN features for repreated sents.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        '-b','--bin_width', dest='bin_widths', nargs='+', type=int, action='store', 
        default=[20],
        # choices=[],
        help="Choose bin_width for normalizers."
    )

    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',#  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec2', 'speech2text', 'deepspeech2',
                'whisper_tiny', 'whisper_small', 'whisper_base', 'whisper_medium'],
        default='wav2letter_modified', 
        help='model to be used for Regression analysis.'
    )
    parser.add_argument(
        '-l','--layers', dest='layer_IDs', nargs='+', type=int, action='store', default=None,
        # choices=[],
        help="Specify list of layer IDs."
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

    compute_and_save_context_dependent_normalizers(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")