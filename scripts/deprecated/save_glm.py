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
from auditory_cortex.io_utils.io import write_lmbdas, cache_glm_parameters, read_cached_glm_parameters
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
    shuffled = args.shuffled
    force_redo = args.force_redo

    if shuffled:
        print("Shuffled pressed...!")
    
    # poisson = True  # always True, for GLM

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

    
    print(f"Running GLM model...")
    # These are the only significant sessions, doing this only for GLM..
    sessions = np.array([180627., 180719., 180720., 180731., 180807., 180808., 180814.,
            190606., 191113., 191115., 191121., 191125., 191206., 191210.,
            200205., 200206., 200207., 200213., 200313.]).astype(int).astype(str)
    # sessions = np.array([200206]).astype(int).astype(str)
    delays_grid_search = False
    # sessions = sessions[12:]
    # sessions = sessions[:12]
    # sessions = sessions[:8]
    # sessions = sessions[8:14]

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

                norm = np.zeros(obj.dataloader.get_num_channels(session=session))
                for N_sents in dataset_sizes:
                    # to keep the implementation generic for multiple layer_IDs
                    glm_coefficients_list = []
                    natural_param_list = []
                    existing_results = read_cached_glm_parameters(
                        model_name, session, bin_width=bin_width, shuffled=shuffled)   
                    for layer_ID in args.layer_IDs:
                        if force_redo or (existing_results is None) or (layer_ID not in existing_results.keys()):
                            neural_spikes, natural_param, glm_coefficients, optimal_lmbdas, lmbda_loss = obj.cross_validated_regression(
                                    session, bin_width=bin_width, delay=delay, iterations=1,
                                    num_folds=k_folds_validation, N_sents=N_sents, return_dict=False,
                                    numpy=use_cpu,third=third, layer_IDs=[layer_ID],
                                    poisson=True, shuffled=shuffled
                                )
                            
                            cache_glm_parameters(model_name, layer_ID, session,
                                neural_spikes, glm_coefficients, natural_param, bin_width=bin_width,
                                shuffled=shuffled
                                )
                            write_lmbdas(
                                model_name, layer_ID, session, optimal_lmbdas, lmbda_loss, bin_width=bin_width
                                )
                        else:
                            neural_spikes = existing_results[layer_ID]['neural_spikes']
                            glm_coefficients = existing_results[layer_ID]['glm_coefficients']
                            natural_param = existing_results[layer_ID]['natural_param']

                        glm_coefficients_list.append(glm_coefficients)
                        natural_param_list.append(natural_param)
                    
                    glm_coefficients = np.concatenate(glm_coefficients_list, axis=0)    # (layer, num_features, ch)
                    natural_param = np.concatenate(natural_param_list, axis=2)          # (samples, ch, layer)
                                
                    corr_coeff = utils.cc_norm(neural_spikes, np.exp(natural_param))
                    poiss_entropy = utils.poisson_cross_entropy(neural_spikes, natural_param)
                
                    Nsamples, Nchannels = neural_spikes.shape
                    data_sums = np.sum(neural_spikes, axis=0, keepdims=True)
                    data_means = data_sums/Nsamples
                    # using mean spikes as predicted means
                    poiss_entropy_baseline = utils.poisson_cross_entropy(neural_spikes, np.log(data_means[...,None]))
                    uncertainty_per_spike = poiss_entropy/data_sums.T/np.log(2)
                    bits_per_spike_NLB = (poiss_entropy_baseline - poiss_entropy)/data_sums.T/np.log(2)


                    corr_dict = {'test_cc_raw': corr_coeff.transpose((1,0)),
                        'train_cc_raw': corr_coeff.transpose((1,0)),
                        'win': bin_width,
                        'delay': delay, 
                        'session': session,
                        'model': obj.model_name,
                        'N_sents': N_sents,
                        'layer_ids': args.layer_IDs,
                        'opt_delays': None,
                        'poiss_entropy': poiss_entropy.transpose((1,0)),
                        'uncertainty_per_spike': uncertainty_per_spike.transpose((1,0)),
                        'bits_per_spike_NLB': bits_per_spike_NLB.transpose((1,0)),
                        }
          
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
    parser.add_argument(
        '-s','--shuffle', dest='shuffled', action='store_true', default=False,
        # choices=[],
        help="Specify if shuffled network to be used."
    )
    parser.add_argument(
        '-f','--force', dest='force_redo', action='store_true', default=False,
        # choices=[],
        help="Force redoing the glm regression."
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