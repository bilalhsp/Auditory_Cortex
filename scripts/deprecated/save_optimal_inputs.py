import os
import argparse
import numpy as np
from scipy.io import wavfile


# local
from auditory_cortex import opt_inputs_dir
from auditory_cortex.analyses.deprecated.regression_correlations import Correlations
from auditory_cortex.optimal_input import OptimalInput
 


# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description="Compute optimal inputs for neurons (ch) from \
            specific recoding session (session), using model (model_name).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
                    )

    # add arguments to read from command line
    parser.add_argument(
        'model_name', action='store',#  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec', 'speech2text'], 
        help='model to be used for synthesizing input.'
    )
    parser.add_argument(
        '--sessions', dest='sessions', type=int, nargs='+', action='store', default=[200206],
        help="List of session IDs for running the analysis on."
    )
    parser.add_argument(
        '-s','--sent', dest='starting_sent', type=int, action='store', default=12,# choices=range(0, 5),
        help="starting sent ID in range(0, 500), '0' means start from random noise."
    )
    parser.add_argument(
        '-t', '--threshold', dest='threshold', type=float, action='store', default=0.25, 
        help="correlation coefficients threshold to select neuron channels"
    )
    parser.add_argument(
        '-l','--layers', dest='layers', type=int, nargs='+', action='store', default=range(3,10),
        help="Layer IDs to run the analysis for."
    )
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False, 
        help="set to True to recompute already existing optimal input"
    )

    return parser




# ------------------  Generate and save optimal inputs ----------------------#

def generate_optimal_inputs(args):

    model_name = args.model_name
    sessions = args.sessions
    start_sent = args.starting_sent
    threshold = args.threshold
    layers = args.layers
    force_redo = args.force_redo

    opt_obj = OptimalInput(model_name, load_features=True)
    corr_obj = Correlations(model_name)

    # if 'None' do it for all sessions
    if sessions is None:
        sessions = corr_obj.get_significant_sessions(threshold=threshold)

    for session in sessions:
        layer_channels_done = []
        # check if the directory already exists or not.
        sub_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles', str(int(session)))
        if not os.path.exists(sub_dir):
            print(f"Creating directory: {sub_dir}")
            os.makedirs(sub_dir)
        else:
            filenames = os.listdir(sub_dir)
            for filename in filenames:
                ind = filename.rfind('.wav')
                starting_sent = int(filename[ind-3:ind])
                if starting_sent == start_sent:    
                    ind = filename.rfind('_corr_')
                    layer_channels_done.append(filename[ind-5:ind])
        
        channels = corr_obj.get_good_channels(session, threshold=threshold)
        for layer in layers:
            print(f"For layer-{layer}")
            for ch in channels:
                print(f"\t ch-{ch}", end='\t')

                if f'{layer:02.0f}_{ch:02.0f}' not in layer_channels_done or force_redo:                
                    inputs, losses, basic_loss, TVloss, grads = opt_obj.get_optimal_input(
                            session=session, layer=layer, ch=ch, starting_sent=start_sent,
                        )
                    corr = corr_obj.get_corr_score(session, layer, ch)
                    # saving the optimal
                    optimal = np.array(inputs[-1].squeeze())
                    filename = f"optimal_{model_name}_{session}_{layer:02.0f}_{ch:02.0f}_corr_{corr:.2f}_starting_{start_sent:03d}.wav"
                    path = os.path.join(sub_dir, filename)
                    wavfile.write(path, 16000, optimal.astype(np.float32))
                    print(f"Saved optimal wavefile for, layer: {layer}, ch: {ch}...!")
                
                else:
                    print(f"optimal wavefile already exists.")


# --------------------------------  main() ---------------------------------#


def main():

    # parge the arguments
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    # generate optimal inputs and save to the disk.
    generate_optimal_inputs(args)



if __name__ == "__main__":
    main()


