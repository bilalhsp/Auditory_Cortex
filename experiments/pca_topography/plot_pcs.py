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
from auditory_cortex.analysis.config import *
import auditory_cortex.helpers as helpers
import auditory_cortex.analysis.analysis as analysis
from utils_jgm.tikz_pgf_helpers import tpl_save
from sklearn.decomposition import PCA


# def plot_select_pcs(pcs, spikes, session=200206, layer=8, threshold=0.4, levels=None, clrm='plasma'):

#     if levels is None:
#         levels = [0.7, 0.75, 0.8]
#     fig, ax = plt.subplots(figsize=(10,10))
#     cmap = mpl.cm.get_cmap(clrm)
    
#     channels = pca_obj.get_good_channels(session, layer, threshold)
#     # fig, ax = plt.subplots()
#     N = len(channels)
#     legend_elements = []
#     for i, ch in enumerate(channels):    
#         # cs = pca_obj.plot_kde(session, layer, ch, ax=ax, levels=levels, color=cmap(i/N))
#         weights = spikes[:,int(ch)]
#         z, *extent = get_kde(pcs, weights)
#         plot_kde(z, extent, ax, levels=levels, color=cmap(i/N))

#         cc = pca_obj.get_corr_score(session, layer, ch)
#         legend_elements.append(Line2D([0], [0], color=cmap(i/N), lw=4, 
#             label=f'ch-{ch}, \u0393-{cc:.2f}'))
    
#     plt.title(f"All good channels for session-{session}, layer-{layer}", fontsize=12)
#     # cax = plt.axes([0.95, 0.2, 0.04, 0.6])
#     # mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=sorted(channels))

#     ax.legend(handles=legend_elements, loc='best')

# def get_kde(pcs, weights):

#     # 100 points on both axis
#     x_min = pcs[0].min()
#     x_max = pcs[0].max()
#     y_min = pcs[1].min()
#     y_max = pcs[1].max()
#     X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
#     positions = np.vstack([X.ravel(), Y.ravel()])
#     # print(pcs.shape)
#     # print(weights.shape)
#     # creating gaussian_kde object and getting values...
#     kernel = scp.stats.gaussian_kde(dataset=pcs, weights=weights)
#     values = kernel(positions)
#     z = np.reshape(values, X.shape).T

#     return z, x_min, x_max, y_min, y_max

# def plot_kde(z, extent, ax, levels, color):
#     x_min, x_max, y_min, y_max = extent
#     colors = [color]
#     X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
#     # print("For this selection: xmin, xmax, ymin, ymax are:")
#     # print(extent)
#     cs = ax.contour(X.T,Y.T,z/z.max(), levels=levels, colors=colors,\
#         #extent= extent
#         )



saved_results = '/home/ahmedb/projects/Wav2Letter/saved_results/'
pretrained_dir = '/depot/jgmakin/data/auditory_cortex/pretrained_weights/w2l_modified/'
checkpoint_file = 'Wav2letter-epoch=024-val_loss=0.37.ckpt'
checkpoint = os.path.join(pretrained_dir, checkpoint_file)
pca_obj = analysis.PCA_topography(checkpoint=checkpoint)

# pca_obj.reg_obj.load_features_and_spikes(bin_width = 20, load_raw=True)

# features = pca_obj.reg_obj.features
# spikes = pca_obj.reg_obj.spikes
# layer = 8

# # # Finding pc space 
# pca = PCA(n_components=10)
# pc_10 = pca.fit_transform(features[layer]).transpose()

# 
session = 200206
# layer = 8
layers = [6,7]
levels = [0.9]
corr_sign_threshold = 0.4
margin=0.8
normalized = True
clrm = 'tab20'
threshold_factor=100
if normalized:
    extend_name = '_norm'
else:
    extend_name = ''
for layer in layers:
    print(f"Saving normalized kde for layer: {layer}")
    for i in range(0, 4):
        for j in range(1,4):
            if i == j or i>j:
                continue
            pc_ind = [i,j]
            print(f" Running for pcs: {pc_ind}")
            # plot_select_pcs(pc_10[pc_ind], spikes, levels=[0.9])
            ax = pca_obj.plot_good_channels_for_session_and_layer(session, layer, levels=levels,
                                                        corr_sign_threshold=corr_sign_threshold, 
                                                        comps=pc_ind, threshold_factor=threshold_factor,
                                                        margin=margin, normalized=normalized, clrm=clrm)
            ax.set_title(f"PCs-{pc_ind}, layer-{layer}, session-{session}{extend_name}_thres_factor_{threshold_factor}")
            plt.savefig(f'../../../saved_results/pcs/normalized/layer_{layer}_pc{pc_ind}{extend_name}_{threshold_factor}.jpg')
            # plt.savefig(f'./saved_results/pcs/layer_{layer}_pc{pc_ind}_{extend_name}.jpg')


