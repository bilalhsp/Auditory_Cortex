import time
import argparse
from auditory_cortex.analyses.deprecated.rsa import RSA 


# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save RSA matrices for layers '+
            'of Regression models and neural areas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # add arguments to read from command line
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',#  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec2', 'speech2text', 'deepspeech2',
                'whisper_tiny', 'whisper_small', 'whisper_base', 'whisper_medium'],
        default='wav2letter_modified', 
        help='model to be used for RSA analysis.'
    )
    parser.add_argument(
        '-n', '--neural', dest='neural', action='store_true', default=False,
        help="RSA for neural data.."
    )
    parser.add_argument(
        '-k','--keys', dest='keys', nargs='+', action='store', default=None,
        help="List of layer ID's of Neural areas."
    )
    parser.add_argument(
        '-i','--identifier', dest='identifier', type=str, action='store', default='',
        choices=['', 'global', 'average'],
        help="Identifier to choose what happens with time axis."
    )
    parser.add_argument(
        '-b','--bin_width', dest='bin_widths', nargs='+', type=int, action='store', 
        default=[20],
        # choices=[],
        help="Choose bin_width for RSA of neural data."
    )
    parser.add_argument(
        '-f', '--force_redo',dest='force_redo', action='store_true', default=False,
        help='Force redo RSA for the current configuration.'
    )
    return parser


# ------------------  RSA computing function ----------------------#

def compute_and_save_RSA(args):

    # create an RSA object for the model_name
    rsa = RSA(model_name=args.model_name, identifier=args.identifier)

    if args.keys is None:
        if args.neural:
            keys = ['all', 'core', 'belt']
        else:
            keys = rsa.model.layer_ids
    else:
        keys = args.keys


    # all keys in the list
    for bin_width in args.bin_widths:
        bin_width = int(bin_width)
        print(f"Saving RDMs for bin_width={bin_width}")
        for key in keys:
            matrix = rsa.get_RDM(
                key=key, neural=args.neural, force_redo=args.force_redo,
                bin_width=bin_width
                )
            # if args.neural:
            #     spikes_belt = rsa.get_neural_spikes(bin_width=bin_width, area=key)
            # else:
            #     feats = rsa.get_layer_features(key, bin_width=bin_width)


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_RSA(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")