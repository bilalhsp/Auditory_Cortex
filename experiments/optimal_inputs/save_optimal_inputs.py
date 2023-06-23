import os
import pandas as pd
from PIL import Image
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
from utils_jgm.tikz_pgf_helpers import tpl_save
import auditory_cortex.analysis.analysis as analysis


# results_dir = '/depot/jgmakin/data/auditory_cortex/correlation_results/cross_validated_correlations'
# sub_dir = 'optimal_inputs'
# saved_optimals = os.path.join(results_dir, sub_dir)
saved_optimals = '/depot/jgmakin/data/auditory_cortex/correlation_results/cross_validated_correlations/optimal_inputs/saved_wavefiles'

import auditory_cortex.optimal_input as op_inp
obj = op_inp.optimal_input('wave2letter_modified', load_features=True)


corr_obj = analysis.correlations()

layer = 6
ch= 30
session = 200206
sent=0
threshold = 0.45
saved_optimals = '/depot/jgmakin/data/auditory_cortex/correlation_results/cross_validated_correlations/optimal_inputs/saved_wavefiles'

channels = corr_obj.get_good_channels(session, layer=layer, threshold=threshold)
for ch in channels:
    ch = int(ch)
    inputs, losses, basic_loss, TVloss, grads = obj.optimize(
        session=session, layer=layer, ch=ch, starting_sent=sent,
        )

    # saving the optimal
    optimal = np.array(inputs[-1].squeeze())
    filename = os.path.join(saved_optimals, f"optimal_{layer}_{ch}_starting_{sent}.wav")
    wavfile.write(filename, 16000, optimal.astype(np.float32))
    print(f"Saved optimal wavefile for, layer: {layer}, ch: {ch}...!")
