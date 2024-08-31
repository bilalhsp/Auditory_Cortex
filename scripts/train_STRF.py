import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV

# local
from auditory_cortex.neural_data import NeuralMetaData
from auditory_cortex import utils, config, saved_corr_dir
from auditory_cortex.io_utils.io import write_model_parameters
from auditory_cortex.computational_models import baseline



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
    lags = [300] #[5, 10, 20, 40, 80, 160, 320]
    test_trial=None
    use_nonlinearity=args.non_linearity
    mVocs = args.mVocs
    mel_spectrogram = args.mel_spectrogram

    if identifier != '':
        identifier = identifier + '_'
    
    csv_file_name = f'STRF_freqs{num_freqs}_{identifier}corr_results.csv'
    # if third is None:
    #     csv_file_name = f'STRF_freqs{num_freqs}_corr_results.csv'
    # else:
    #     csv_file_name = f'STRF_{third}_third_corr_results.csv'
    # CSV file to save the results at
    file_exists = False
    file_path = os.path.join(saved_corr_dir, csv_file_name)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        file_exists = True

    metadata = NeuralMetaData()
    sessions = metadata.get_all_available_sessions()
    # ## read the sessions available in data_dir
    # sessions = np.array(os.listdir(data_dir))
    # sessions = np.delete(sessions, np.where(sessions == "out_sentence_details_timit_all_loudness.mat"))
    # for s in bad_sessions:
    #     sessions = np.delete(sessions, np.where(sessions == s))
    # sessions = np.sort(sessions)

    sessions = sessions[args.start_ind:args.end_ind]

    if file_exists:
        sessions_done = data[(data['delay']==delay) & (data['bin_width']==bin_width)]['session'].unique()
        subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
    else:
        subjects = sessions

    strf_model = baseline.STRF(mel_spectrogram=mel_spectrogram)
    # subjects = np.array(['200206'])
    for session in subjects:
        if mVocs:
            excluded_sessions = ['190726', '200213']
            if session in excluded_sessions:
                print(f"Excluding session: {session}")
                continue
        print(f"\n Working with '{session}'")
        # obj = get_reg_obj(data_dir, sub)

        corr, opt_lag, opt_lmbda = strf_model.grid_search_CV(
            session, bin_width, lags=lags, tmin=tmin,
            num_workers=num_workers, num_lmbdas=num_alphas, 
            num_folds=num_folds, use_nonlinearity=use_nonlinearity,
            test_trial=test_trial, mVocs=mVocs
        )


        # Deprecated
        # alphas = np.logspace(-2, 5, num_alphas)
        # estimator = RidgeCV(alphas=alphas, cv=5)

        # strf_model = baseline.STRF(
        #             session,
        #             estimator=estimator,
        #             num_workers=num_workers, 
        #             num_freqs=num_freqs,
        #             tmin=tmin,
        #             tmax=tmax,
        #             bin_width=bin_width
        #         )

        # corr = strf_model.fit(third=third)
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

        # writing coefficients...
        # coeff = strf_model.get_coefficients()

        # write_model_parameters(strf_model.model_name, session, coeff)

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save WER for pretrained models ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    # parser.add_argument(
    #     '-l','--lags', dest='lags', nargs='+', type=int,
    #     action='store', default=[80],
    #     help="Specify the maximum lag used for STRF."
    # )
    # parser.add_argument(
    #     '--tmin', dest='tmin', type=float, action='store', default=0.0,
    #     help="Specify the minimum lag used for STRF."
    # )
    parser.add_argument(
        '-s','--start', dest='start_ind', type=int, action='store', 
        default=0,
        # choices=[],
        help="Choose sessions starting index to compute results at."
    )
    parser.add_argument(
        '-e','--end', dest='end_ind', type=int, action='store', 
        default=41,
        # choices=[],
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




       
