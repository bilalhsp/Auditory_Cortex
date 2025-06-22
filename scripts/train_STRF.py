import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
import auditory_cortex.io_utils.io as io

# local

from auditory_cortex import utils, config, saved_corr_dir
from auditory_cortex.io_utils.io import write_model_parameters
from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata
from auditory_cortex.data_assembler import STRFDataAssembler
from auditory_cortex.encoding import TRF


# ------------------  Baseline computing function ----------------------#

def compute_and_save_STRF_baseline(args):
    data_dir = config['neural_data_dir']
    bad_sessions = config['bad_sessions']
    # results_dir = config['results_dir']

    bin_width = args.bin_width
    tmin = 0
    # tmax = args.tmax
    identifier = args.identifier
    # sfreq = 100

    num_freqs = 80
    num_workers = 6
    delay = 0.0
    num_alphas = 8
    num_folds=3
    third = None
    lags = [args.lag] #[5, 10, 20, 40, 80, 160, 320]
    test_trial=None
    use_nonlinearity=args.non_linearity
    mVocs = args.mVocs
    mel_spectrogram = args.mel_spectrogram
    dataset_name = args.dataset_name
    spectrogram_type = args.spectrogram_type

    results_identifier = utils.get_run_id(
            dataset_name, bin_width, identifier, mVocs=mVocs, lag=lags[0],
        )
    if mel_spectrogram:
        if spectrogram_type is None or 'speech2text' in spectrogram_type:
            substr = 'mel_'
        elif 'whisper' in spectrogram_type:
            substr = 'mel_wh_'
        elif 'deepspeech2' in spectrogram_type:
            substr = 'mel_ds_'
        elif 'librosa' in spectrogram_type:
            substr = 'mel_lib_'
        else:
            raise ValueError(f"Unknown spectrogram type: {spectrogram_type}")
        results_identifier = substr + results_identifier
    else:
        if spectrogram_type is None or 'wavlet' in spectrogram_type:
            substr = 'wavlet_'
        elif 'cochleogram' in spectrogram_type:
            substr = 'coch_'
        results_identifier =  substr + results_identifier

    logging.info(f"Results identifier: {results_identifier}")
    csv_file_name = f'STRF_freqs{num_freqs}_{results_identifier}_corr_results.csv'

    # CSV file to save the results at
    file_exists = False
    file_path = os.path.join(saved_corr_dir, csv_file_name)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        file_exists = True

    metadata = create_neural_metadata(dataset_name)
    sessions = metadata.get_all_available_sessions()
    sessions = sessions[args.start_ind:args.end_ind]

    if file_exists:
        sessions_done = data[(data['delay']==delay) & (data['bin_width']==bin_width)]['session'].unique()
        subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
    else:
        subjects = sessions

    dataset_obj = create_neural_dataset(dataset_name, subjects[0])
    dataset = STRFDataAssembler(
        dataset_obj, bin_width, mVocs=mVocs,
        mel_spectrogram=mel_spectrogram,
        spectrogram_type=spectrogram_type,
        )

    for session in subjects:
        if mVocs:
            excluded_sessions = ['190726', '200213']
            if session in excluded_sessions:
                print(f"Excluding session: {session}")
                continue
        logging.info(f"\n Working with '{session}'")

        if session != dataset.dataloader.dataset_obj.sub:
            # no need to read features again...just reach spikes..
            dataset_obj = create_neural_dataset(dataset_name, session)
            dataset.read_session_spikes(dataset_obj)
            # dataset = STRFDataAssembler(
            #     dataset_obj, bin_width, mVocs=mVocs,
            #     mel_spectrogram=mel_spectrogram,
            #     spectrogram_type=spectrogram_type,
            #     )
        model_name = 'strf'
        trf_obj = TRF(model_name, dataset)
        
        corr, opt_lag, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                lags=lags, tmin=tmin,
                num_folds=num_folds,
                test_trial=test_trial
            )

        # betas = trf_model.coef_
        # io.write_trf_parameters(
        #     model_name, session, betas, bin_width=bin_width,
        #     )
        if mVocs:
            mVocs_corr = corr
            timit_corr = np.zeros_like(corr)
        else:
            mVocs_corr = np.zeros_like(corr)
            timit_corr = corr

        results_dict = {
            'win': bin_width,
            'delay': delay,
            'session': session,
            'strf_corr': timit_corr,
            'mVocs_strf_corr': mVocs_corr,
            'num_freqs': num_freqs,
            'tmin': tmin,
            'tmax': opt_lag,
            'lmbda': np.log10(opt_lmbda),
            }
        df = utils.write_STRF(results_dict, file_path)


# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save WER for pretrained models ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '--lag', dest='lag', type=int, action='store', 
        default=200,
        help="Specify the maximum lag used for STRF."
    )
    # parser.add_argument(
    #     '--tmin', dest='tmin', type=float, action='store', default=0.0,
    #     help="Specify the minimum lag used for STRF."
    # )
    parser.add_argument(
        '-d','--dataset_name', dest='dataset_name', type= str, action='store',
        choices=['ucsf', 'ucdavis'],
        help = "Name of neural data to be used."
    )
    parser.add_argument(
        '-s','--start', dest='start_ind', type=int, action='store', 
        default=0,
        help="Choose sessions starting index to compute results at."
    )
    parser.add_argument(
        '-e','--end', dest='end_ind', type=int, action='store', 
        default=41,
        help="Choose sessions ending index to compute results at."
    )
    parser.add_argument(
        '-i','--identifier', dest='identifier', type= str, action='store',
        default='',
        help="Specify identifier for saved results."
    )
    parser.add_argument(
        '-b','--bin_width', dest='bin_width', type= int, action='store', default=50,
        help="Specify the bin_width to use for analysis."
    )
    parser.add_argument(
        '-n','--non_linearity', dest='non_linearity', 
        action='store_true', default=False,
        help="Use non-linearity after the linear model."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if spikes for mVocs are to be used."
    )
    parser.add_argument(
        '--mel', dest='mel_spectrogram', action='store_true', default=False,
        help="Specify if mel_spectrogram to be used as baseline."
    )
    parser.add_argument(
        '--type', dest='spectrogram_type', type= str, action='store',
        help="Specify the type of spectrogram to be used as baseline."
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

    compute_and_save_STRF_baseline(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")




       
