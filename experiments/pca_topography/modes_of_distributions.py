import os
import numpy as np
import pandas as pd
import scipy as scp
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.lines import Line2D

# 
import auditory_cortex.analysis.config as config
import auditory_cortex.helpers as helpers
import auditory_cortex.analysis.analysis as analysis
from utils_jgm.tikz_pgf_helpers import tpl_save
from sklearn.decomposition import PCA


# saved_results = '/home/ahmedb/projects/Wav2Letter/saved_results/'
# pretrained_dir = '/depot/jgmakin/data/auditory_cortex/pretrained_weights/w2l_modified/'
# checkpoint_file = 'Wav2letter-epoch=024-val_loss=0.37.ckpt'
# checkpoint = os.path.join(pretrained_dir, checkpoint_file)
checkpoint = config.regression_object_paths['saved_checkpoint']
pca_obj = analysis.PCA_topography(checkpoint=checkpoint)


# results_dir = config.results_dir 
file_path = os.path.join(config.results_dir, config.pca_kde_sub_dir, config.pca_dist_modes_filename)
layers = [8, 9, 10, 11]
comps = [0,1,2,3,4]
threshold = 0.3
sessions = pca_obj.get_significant_sessions()

for layer in layers:

    for i, session in enumerate(sessions):
        channels = pca_obj.get_good_channels(session, layer, threshold)
        for k, ch in enumerate(channels):
            for comp in comps:
                pca_obj.save_mode_of_marginal_dist(file_path, session, layer=layer, ch=ch, comp=comp)

print("Done with all significant sessions and channels...")