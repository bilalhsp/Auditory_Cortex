import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV, ElasticNetCV

import naplib as nl
from naplib.visualization import imSTRF

import auditory_cortex.helpers as helpers
import auditory_cortex.analysis.config as config
import auditory_cortex.analysis.analysis as analysis
import auditory_cortex.utils as utils

import os
import time
import yaml
import pandas as pd


start_time = time.time()
print("Starting out...")


reg_conf = '/home/ahmedb/projects/Wav2Letter/Auditory_Cortex/conf/regression_w2l.yaml'
with open(reg_conf, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_dir = config['data_dir']
bad_sessions = config['bad_sessions']
results_dir = config['results_dir']
# delays = config['delays']
# bin_widths = config['bin_widths']
# pretrained = config['pretrained']
# k_folds_validation = config['k_folds_validation']
# iterations = config['iterations']
# use_cpu = config['use_cpu']
# dataset_sizes = config['dataset_sizes']
# dataset_sizes = np.arange(dataset_sizes[0], dataset_sizes[1], dataset_sizes[2])

delay = 0.0
bin_width = 20
num_alphas = 5



csv_file_name = 'STRF_corr_results.csv'
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


# sessions = sessions[30:38]


if file_exists:
    sessions_done = data[(data['delay']==delay) & (data['bin_width']==bin_width)]['session'].unique()
    subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
else:
    subjects = sessions

# subjects = subjects[2:]

for session in subjects:
    print(f"Working with '{session}'")
    # obj = get_reg_obj(data_dir, sub)
    strf = analysis.STRF(session)
    # norm = obj.get_normalizer(session, bin_width=bin_width, delay=delay)
    # for N_sents in dataset_sizes:
    # choose an estimator...
    ridge = True
    # estimator = Ridge(alpha=1.0, max_iter=10)
    # estimator = ElasticNet(l1_ratio=0.01)
    # estimator=None
    if ridge:
        alphas = np.logspace(-2, 5, num_alphas)
        estimator = RidgeCV(alphas=alphas, cv=5)
        # filename = 'STRF_corr_RidgeCV'
    else:
        alphas = np.logspace(-2,5, num_alphas)
        estimator = ElasticNetCV()
        # filename = 'STRF_corr_elasticNetCV'

    strf_model, corr = strf.fit(estimator, num_workers=4)

    results_dict = {
        'win': bin_width,
        'delay': delay,
        'session': session,
        'strf_corr': corr
        }
    df = utils.write_STRF(results_dict, file_path)

    
END = time.time()
print(f"Took {(END-start_time)/60:.2f} min., for bin_widths: '{bin_width}' and delays: '{delay}'.")
    
    
    
    
    
    
#     corr_dict = obj.cross_validated_regression(session, bin_width=bin_width, delay=delay,
#                 N=iterations, k=k_folds_validation, N_sents=N_sents,
#                 return_dict=True, numpy=use_cpu)
#     df = utils.write_to_disk(corr_dict, file_path, normalizer=norm)


# # creating object of my STRF class..
# strf = analysis.STRF()


# current_time = time.time()
# elapsed_time = current_time - start_time
# print(f"Model created in {elapsed_time:.3f}")
# print("Training the model...")

# # choose an estimator...
# ridge = True

# # estimator = Ridge(alpha=1.0, max_iter=10)
# # estimator = ElasticNet(l1_ratio=0.01)
# # estimator=None
# if ridge:
#     alphas = np.logspace(-2,5, 8)
#     estimator = RidgeCV(alphas=alphas, cv=5)
#     filename = 'STRF_corr_RidgeCV'
# else:
#     # alphas = np.logspace(-2,5, 8)
#     estimator = ElasticNetCV()
#     filename = 'STRF_corr_elasticNetCV'

# strf_model, corr = strf.fit(estimator, num_workers=4)


# current_time = time.time()
# elapsed_time = current_time - start_time
# print(f"Model trained, it took {elapsed_time:.3f}")
# print(corr.shape)
# print(f"Correlation coefficient for RidgeCV model:", corr)

# # saving correlation results...
# # results = {'200206': corr}
# path = os.path.join(config.results_dir, config.corr_sub_dir, filename)
# np.save(path, corr)#results, allow_pickle=True)