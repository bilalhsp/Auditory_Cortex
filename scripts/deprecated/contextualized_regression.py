import os
import pandas as pd
import soundfile
import yaml
import torchaudio
import scipy
import matplotlib.pyplot as plt
import torch
from scipy.io import wavfile
import numpy as np
import pickle
import time
import argparse

# local
from auditory_cortex import config
import auditory_cortex.utils as utils
import auditory_cortex.deprecated.models as models
# from wav2letter.datasets import DataModuleRF 
# from wav2letter.models import LitWav2Letter, Wav2LetterRF



# reg_conf = '/home/ahmedb/projects/Wav2Letter/Auditory_Cortex/conf/regression_w2l.yaml'
# with open(reg_conf, 'r') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)


def compute_and_save_regression(args):

    START = time.time()
    data_dir = config['neural_data_dir']
    bad_sessions = config['bad_sessions']
    results_dir = config['results_dir']
    results_dir = os.path.join(results_dir, 'cross_validated_correlations')
    delays = config['delays']
    # bin_widths = config['bin_widths']
    bin_widths = args.bin_widths
    # pretrained = config['pretrained']
    k_folds_validation = config['k_folds_validation']
    iterations = config['iterations']
    use_cpu = config['use_cpu']
    dataset_sizes = config['dataset_sizes']
    dataset_sizes = np.arange(dataset_sizes[0], dataset_sizes[1], dataset_sizes[2])

    # model_name = config['model_name']
    model_name = args.model_name

    # identifier = config['identifier']
    identifier = args.identifier
    delay_features = config['delay_features']
    audio_zeropad = config['audio_zeropad']

    delays_grid_search = config['delays_grid_search']
    third = config['third']
    if not third:
        third = None
    # # Create w2l model..





    # use_cpu = True
    # csv_file_name = 'testing_for_modified_code.csv'
    csv_file_name = 'corr_results.csv'
    if identifier != '':
        csv_file_name = identifier + '_' + csv_file_name

    csv_file_name = model_name + '_' + csv_file_name
    # CSV file to save the results at
    file_exists = False
    file_path = os.path.join(results_dir, csv_file_name)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        file_exists = True

    ## read the sessions available in data_dir
    sessions = np.array(os.listdir(data_dir))
    sessions = np.delete(sessions, np.where(sessions == "out_sentence_details_timit_all_loudness.mat"))
    for s in bad_sessions:
        sessions = np.delete(sessions, np.where(sessions == s))
    sessions = np.sort(sessions)

    # sessions = sessions[:20]
    # sessions = sessions[20:]
    # sessions = sessions[15:30]


    obj = models.Regression(
                model_name=model_name, delay_features=delay_features, audio_zeropad=audio_zeropad
            )
    current_time = time.time()
    elapsed_time = current_time - START
    print(f"It takes {elapsed_time:.2f} seconds to load features...!")
    # sents = [12,13,32,43,56,163,212,218,287,308]
    for delay in delays:
        for bin_width in bin_widths:
            # sessions = np.array(['200206'])
            # Session in data_dir that we do not have results for...
            if file_exists:
                sessions_done = data[
                        (data['delay']==delay) & \
                        (data['bin_width']==bin_width) 
                    ]['session'].unique()

                subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
            else:
                subjects = sessions
            
            for session in subjects:
                print(f"Working with '{session}'")
                # obj = get_reg_obj(data_dir, sub)

                # norm = obj.get_normalizer(session, bin_width=bin_width, delay=delay,
                #                         n=1 # normalizer not needed, will be updated later
                #
                ## normalizer not needed, will be updated later, fill in dummy values for now...
                norm = np.zeros(obj.dataloader.get_num_channels(session=session))
                for N_sents in dataset_sizes:
                    if delays_grid_search:
                        # delays_grid = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
                        # delays_grid = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
                        delays_grid = [0,10,20,30,40,50,60,70,80,90,100] # for high bin_widths..
                        # delays_grid = [0] # Only for 1000 bin_width

                        corr_dict = obj.grid_search_CV_contextualized(
                                session, bin_width=bin_width, iterations=iterations,
                                num_folds=k_folds_validation, N_sents=N_sents, return_dict=True,
                                numpy=use_cpu, delays=delays_grid, third=third, layer_IDs=args.layer_IDs
                            )
                    else:
                        corr_dict = obj.cross_validated_regression(
                                session, bin_width=bin_width, delay=delay, iterations=iterations,
                                num_folds=k_folds_validation, N_sents=N_sents, return_dict=True,
                                numpy=use_cpu,third=third, layer_IDs=args.layer_IDs
                            )
                    df = utils.write_to_disk(corr_dict, file_path, normalizer=norm)

    END = time.time()
    print(f"Took {(END-START)/60:.2f} min., for bin_widths: '{bin_widths}' and delays: '{delays}'.")



# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save regression results for for layers '+
            'of DNN models and neural areas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # add arguments to read from command line
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',#  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec2', 'speech2text', 'deepspeech2',
                'whisper_tiny', 'whisper_small', 'whisper_base', 'whisper_medium'],
        default='wav2letter_modified', 
        help='model to be used for Regression analysis.'
    )

    parser.add_argument(
        '-b','--bin_widths', dest='bin_widths', nargs='+', type= int, action='store', default=[20],
        # choices=[],
        help="Specify list of bin_widths."
    )
    parser.add_argument(
        '-l','--layers', dest='layer_IDs', nargs='+', type=float, action='store', default=None,
        # choices=[],
        help="Specify list of layer IDs."
    )
    parser.add_argument(
        '-i','--identifier', dest='identifier', type= str, action='store',
        default='sampling_rate_opt_neural_delay',
        # choices=[],
        help="Specify identifier for saved results."
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

    compute_and_save_regression(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")