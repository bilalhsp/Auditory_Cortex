# adjsut the basic logging lovel of notebook
import logging

from auditory_cortex.analyses.deprecated import regression_correlations
logging.basicConfig(level=logging.WARNING)

import os
import pickle
import numpy as np
from auditory_cortex import optimal_input, opt_inputs_dir
from auditory_cortex import utils


threshold = 0.068
layer_id = 2
model_name = 'deepspeech2'
# model_name = 'wav2letter_modified'

# correlation object
corr_obj = regression_correlations.Correlations(model_name=model_name+'_'+'opt_neural_delay')
opt_obj = optimal_input.OptimalInput(model_name=model_name, load_features=False)

# read betas...
dirpath = os.path.join(opt_inputs_dir, model_name)
filepath = os.path.join(dirpath, f"{model_name}_beta_bank.pkl")
if os.path.exists(filepath):
    with open(filepath, 'rb') as f:
        beta_bank = pickle.load(f)
        print("Loading file...")
else:
    raise FileNotFoundError(f"Results not saved, check and recompute...!")


l = opt_obj.get_layer_index(layer_id)
# sessions = ['200213', '180731']
# sessions_list_vertical = sessions
# sessions_list_horizontal = sessions
sessions_list_vertical = beta_bank.keys()
sessions_list_horizontal = beta_bank.keys()
beta_beta_confusion = []
for session1 in sessions_list_vertical:
    beta_beta_row = []
    channels_1 = corr_obj.get_good_channels(session1 ,threshold=threshold)
    beta1 = beta_bank[session1][l].cpu().numpy()

    for session2 in sessions_list_horizontal:
        beta2 = beta_bank[session2][l].cpu().numpy()
        channels_2 = corr_obj.get_good_channels(session2, threshold=threshold)

        beta_beta = np.zeros((len(channels_1), len(channels_2)))
        for i, ch1 in enumerate(channels_1):

            for j, ch2 in enumerate(channels_2):
                beta_beta[i,j] = utils.cc_single_channel(beta1[:,int(ch1)], beta2[:,int(ch2)])
        beta_beta_row.append(beta_beta)

    beta_beta_row =  np.concatenate(beta_beta_row, axis=1)
    beta_beta_confusion.append(beta_beta_row)

beta_beta_confusion = np.concatenate(beta_beta_confusion, axis=0)


matrix_filepath = os.path.join(dirpath, f"{model_name}_confusion_matrix_plain.npy")
np.save(matrix_filepath, beta_beta_confusion)
print(f"Confusion matrix saved to file: \n {matrix_filepath}")