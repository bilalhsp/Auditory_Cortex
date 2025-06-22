import os
import time
import argparse
import matplotlib.pyplot as plt
from auditory_cortex import opt_inputs_dir, results_dir, cache_dir
from auditory_cortex.analyses import Correlations
from auditory_cortex.plotters.tikzplots import plot_spectrogram_spike_count_pair


# ------------------  save qualitative figures ----------------------#

def save_qualititive_plots(args):
    """Saves qualitative neural predictions figure, to cache dir
    
    """
    model_name = args.model_name
    bin_width = args.bin_width
    threshold = args.threshold
    sent_ids = args.sent_ids
    layer = args.layer 

    figs_dir = os.path.join(cache_dir, 'qualitative_plots', model_name)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    identifier = 'ucsf_timit_trf_lags200_bw50_regression_improved'
    corr_obj = Correlations(model_name=model_name+'_'+ identifier)
    # sessions = corr_obj.get_significant_sessions(bin_width=bin_width, threshold=threshold)
    sessions = ['180731', '200206']
    for session in sessions:
        session = int(session)
        channels = corr_obj.get_good_channels(session, threshold=threshold, bin_width=bin_width)
        for ch in channels:
            ch = int(ch)
            for sent_id in sent_ids:
                identifier = f'sent{sent_id}-session-{session}-ch{ch}-{bin_width}ms-{model_name}'
                filename = os.path.join(figs_dir, f"{identifier}.jpg")
                if not os.path.exists(filename):

                    plot_spectrogram_spike_count_pair(
                        model_name=model_name, 
                        sent_id=sent_id,
                        session=session,
                        ch=ch,
                        layer=layer,
                        bin_width=bin_width,
                        save_tikz=False
                    )
                    plt.title(identifier)
                    plt.savefig(filename)
                    print(f"Saved: {filename}")
        

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to save the qualitative neural prediction figures to cache dir'+
         'pick the better looking figures for the paper. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-t', '--threshold', dest='threshold', type=float, action='store', default=0.5,
        help="Specify the threshold to pick the channels."
    )
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',#  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec2', 'speech2text', 'deepspeech2',
                'whisper_tiny', 'whisper_small', 'whisper_base', 'whisper_medium',
                'wav2letter_spect',
                ],
        default='whisper_base', 
        help='model to be used for plotting.'
    )
    parser.add_argument(
        '-s','--sent_ids', dest='sent_ids', nargs='+', type= int, action='store',
        default=[12,13,32,43,56,163,212,218,287,308],
        help="Specify list of sent_ids to evaluate."
    )
    parser.add_argument(
        '-b','--bin_width', dest='bin_width', type= int, action='store', default=50,
        help="Specify the bin_width to use for predictions."
    )
    parser.add_argument(
        '-l','--layer', dest='layer', type= int, action='store', default=None,
        help="Specify the layer to be used for predictions."
    )
    return parser




# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    print("Starting out...")
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    save_qualititive_plots(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")




       
