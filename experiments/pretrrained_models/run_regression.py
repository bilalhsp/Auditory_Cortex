import os
import pandas as pd
import soundfile
import yaml
from wav2letter.datasets import Dataset, LSDataModule, DataModuleRF
from wav2letter.models import LitWav2Letter, Wav2LetterRF
import torchaudio
import scipy
import matplotlib.pyplot as plt
import torch
from scipy.io import wavfile
import auditory_cortex.regression as Reg
import auditory_cortex.utils as utils
import numpy as np
import pickle
import time

START = time.time()

reg_conf = '/home/ahmedb/projects/Wav2Letter/Auditory_Cortex/conf/regression_w2l.yaml'
with open(reg_conf, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



data_dir = config['data_dir']
bad_sessions = config['bad_sessions']
results_dir = config['results_dir']
delays = config['delays']
bin_widths = config['bin_widths']
pretrained = config['pretrained']
k_folds_validation = config['k_folds_validation']
iterations = config['iterations']
use_cpu = config['use_cpu']
dataset_sizes = config['dataset_sizes']
dataset_sizes = np.arange(dataset_sizes[0], dataset_sizes[1], dataset_sizes[2])

# # Create w2l model..
# if pretrained:
# # Create model with pretrained weights....!
#     # checkpoint_file = "Wav2letter-epoch=024-val_loss=0.37.ckpt"
#     checkpoint_file = config['checkpoint']
#     pretrained_dir = config['pretrained_dir']
#     checkpoint = os.path.join(pretrained_dir, checkpoint_file)
#     mod = Wav2LetterRF.load_from_checkpoint(checkpoint)
#     csv_file_name = config['pretrained_correlations_file']
# else:
#     mod = Wav2LetterRF()
#     csv_file_name = config['untrained_correlations_file']
# csv_file_name = config['pretrained_correlations_file']

# model_name = 'wave2letter_modified'
# model_name = 'wave2vec2'
model_name = 'speech2text'
# model_name = 'whisper'

# csv_file_name = 'testing_for_modified_code.csv'
csv_file_name = 'corr_results.csv'

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

obj = Reg.transformer_regression(model_name=model_name)
current_time = time.time()
elapsed_time = current_time - START
print(f"It takes {elapsed_time:.2f} seconds to load features...!")
# sents = [12,13,32,43,56,163,212,218,287,308]
for delay in delays:
    for bin_width in bin_widths:

        # Session in data_dir that we do not have results for...
        if file_exists:
            sessions_done = data[(data['delay']==delay) & (data['bin_width']==bin_width)]['session'].unique()
            subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
        else:
            subjects = sessions
        subjects = ['200206']
        for session in subjects:
            print(f"Working with '{session}'")
            # obj = get_reg_obj(data_dir, sub)

            norm = obj.get_normalizer(session, bin_width=bin_width, delay=delay)
            for N_sents in dataset_sizes:
                corr_dict = obj.cross_validated_regression(session, bin_width=bin_width, delay=delay,
                            N=iterations, k=k_folds_validation, N_sents=N_sents,
                            load_features=True, return_dict=True, numpy=use_cpu)
                df = utils.write_to_disk(corr_dict, file_path, normalizer=norm)

END = time.time()
print(f"Took {(END-START)/60:.2f} min., for bin_widths: '{bin_widths}' and delays: '{delays}'.")

