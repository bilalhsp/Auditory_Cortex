import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV

# local
from auditory_cortex import STRF, utils, config, saved_corr_dir


start_time = time.time()
print("Starting out...")

data_dir = config['neural_data_dir']
bad_sessions = config['bad_sessions']
# results_dir = config['results_dir']

tmin = 0
tmax = 0.3
sfreq = 100

delay = 0.0
bin_width = 20
num_alphas = 5
ridge = True
third = 3


if third is None:
    csv_file_name = 'STRF_corr_results.csv'
else:
    csv_file_name = f'STRF_{third}_third_corr_results.csv'
# CSV file to save the results at
file_exists = False
file_path = os.path.join(saved_corr_dir, csv_file_name)
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    file_exists = True

## read the sessions available in data_dir
sessions = np.array(os.listdir(data_dir))
sessions = np.delete(sessions, np.where(sessions == "out_sentence_details_timit_all_loudness.mat"))
for s in bad_sessions:
    sessions = np.delete(sessions, np.where(sessions == s))
sessions = np.sort(sessions)

if file_exists:
    sessions_done = data[(data['delay']==delay) & (data['bin_width']==bin_width)]['session'].unique()
    subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
else:
    subjects = sessions

# subjects = subjects[2:]

for session in subjects:
    print(f"Working with '{session}'")
    # obj = get_reg_obj(data_dir, sub)
    strf = STRF.STRF(session)

    if ridge:
        alphas = np.logspace(-2, 5, num_alphas)
        estimator = RidgeCV(alphas=alphas, cv=5)
        # filename = 'STRF_corr_RidgeCV'
    else:
        alphas = np.logspace(-2,5, num_alphas)
        estimator = ElasticNetCV()
        # filename = 'STRF_corr_elasticNetCV'

    strf_model, corr = strf.fit(
        estimator, num_workers=4, tmin=tmin, tmax=tmax, sfreq=sfreq, third=third
        )

    results_dict = {
        'win': bin_width,
        'delay': delay,
        'session': session,
        'strf_corr': corr
        }
    df = utils.write_STRF(results_dict, file_path)

    
END = time.time()
print(f"Took {(END-start_time)/60:.2f} min., for bin_widths: '{bin_width}' and delays: '{delay}'.")
    
