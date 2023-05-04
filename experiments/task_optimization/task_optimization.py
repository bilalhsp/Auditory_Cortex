import os
import pandas as pd
import soundfile
import fnmatch
import yaml
from wav2letter.datasets import Dataset, LSDataModule, DataModuleRF
from wav2letter.models import LitWav2Letter, Wav2LetterRF
import torchaudio
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.io import wavfile
import auditory_cortex.regression as Reg
import auditory_cortex.utils as utils
import numpy as np
import cupy as cp
import pickle
import time
from utils_jgm.tikz_pgf_helpers import tpl_save

import auditory_cortex.analysis.config as config
import auditory_cortex.helpers as helpers
import auditory_cortex.analysis.analysis as analysis

# paths to read checkpoints from and save corr results at...
models_dir = config.task_optimization_paths['model_checkpoints_dir']
saved_corr_results_dir = config.task_optimization_paths['saved_corr_results_dir']

# read checkpoints from the directory..
checkpoints = []
for file_name in os.listdir(models_dir):
    if fnmatch.fnmatch(file_name, 'Wav2letter-epoch*.ckpt'):
        checkpoints.append(file_name)
checkpoints.sort()

# extract 'epochs' and 'loss' information from the checkpoint filenames...
pretrained_mdata = []
for checkpoint in checkpoints:
    # extract epoch number from latest checkpoint...
    upper_index = checkpoint.index('-val')
    lower_index = checkpoint.rfind('=', 0, upper_index) + 1
    epoch = int(checkpoint[lower_index:upper_index])

    # extract loss value from latest checkpoint...
    upper_index = checkpoint.index('.ckpt')
    lower_index = checkpoint.rfind('=', 0, upper_index) + 1
    loss = float(checkpoint[lower_index:upper_index])

    dictt = {'epochs': epoch,
             'loss': loss,
             'checkpoint': checkpoint
            }
    pretrained_mdata.append(dictt)


session = '200206'
bin_width = 20
delay = 0

for i, pretrained in enumerate(pretrained_mdata):
    # filenames and epoch information...
    epochs = pretrained['epochs']
    checkpoint = os.path.join(models_dir, pretrained['checkpoint'])
    corr_file_path = os.path.join(saved_corr_results_dir,
                                f'w2l_epochs={epochs:02d}_session_{session}_corr.csv')

    # creating regression object...
    reg_obj = helpers.get_regression_obj(session, checkpoint=checkpoint, load_features=True)

    corr_results = reg_obj.cross_validated_regression(bin_width=bin_width,
                                                    delay=delay,
                                                    load_features=True,
                                                    return_dict=True,
                                                    numpy=False)

    # write correlation results to disk...
    utils.write_to_disk(corr_results, corr_file_path)




