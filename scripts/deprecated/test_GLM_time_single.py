from auditory_cortex.analyses import Correlations
from auditory_cortex.deprecated.models import Regression
from auditory_cortex.neural_data.neural_meta_data import NeuralMetaData
from auditory_cortex.plotters.correlation_plotter import RegPlotter
from auditory_cortex.deprecated.dataloader import DataLoader
import auditory_cortex.utils as utils

import os
import torch
import numpy as np
import seaborn as sns

import scipy
import jiwer
import cupy as cp
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pylab as plt
import matplotlib as mpl
# %matplotlib inline

model_name = 'wav2letter_modified'
obj = Regression(model_name=model_name)

import time
session = 200206
bin_width = 20
delay = 0
iterations = 1
k_folds_validation = 5
use_cpu = False
layer_IDs = np.arange(12)
sents = None
num_lmbdas = 10
num_folds = 5

if sents is None:
    sents = obj.dataloader.sent_IDs

N_sents = len(sents)


test_sents = obj.dataloader.test_sent_IDs

# this creates a new dataset object and extracts the spikes
session = str(session)
# spikes = self.unroll_spikes(session, bin_width=bin_width, delay=delay)
num_channels = obj.dataloader.get_num_channels(session)
# _ = self.get_neural_spikes(session, bin_width=bin_width, delay=delay)
# num_channels = self.spike_datasets[session].num_channels
if layer_IDs is None:
    layer_IDs = obj.get_layer_IDs()

# loading features for any one sent=12, to get feature_dims..
feature_dims = obj.unroll_features(
    bin_width=bin_width, sents=[12], layer_IDs=layer_IDs).shape[-1]

# feature_dims = self.sampled_features[0].shape[1]
lmbdas = np.logspace(start=-4, stop=-1, num=num_lmbdas)
B = np.zeros((len(layer_IDs), feature_dims, num_channels))
corr_coeff = np.zeros((iterations, num_channels, len(layer_IDs)))
poiss_entropy = np.zeros((iterations, num_channels, len(layer_IDs)))
# corr_coeff_train = np.zeros((iterations, num_channels, len(layer_IDs)))
# stimuli = np.array(list(self.raw_features[0].keys()))

stimuli = np.random.permutation(sents)[0:N_sents]
# mapping_sents = int(N_sents*0.7) # 70% test set...!
# size_of_chunk = int(mapping_sents/k)
print(f"# of iterations requested: {iterations}, \n \
        # of lambda samples per iteration: {len(lmbdas)}")
time_itr = 0
time_lmbda = 0
time_map = 0
# time_fold = 0

start_itr = time.time()

np.random.shuffle(stimuli)

# if test_sents is None:
#     mapping_set = stimuli[:mapping_sents]
#     test_set = stimuli[mapping_sents:]
# else:
# option to fix the test set..!
mapping_set = stimuli[np.isin(stimuli, test_sents, invert=True)]
test_set = test_sents

# lmbda_loss = module.zeros(((len(lmbdas), num_channels, self.num_layers)))
start_lmbda = time.time()

# K-fold CV - GLIM...
print(f"{num_folds}_fold CV for session: {session}")
# num_channels = self.spike_datasets[session].num_channels
num_channels = obj.dataloader.get_num_channels(session)
if layer_IDs is None:
    layer_IDs = obj.get_layer_IDs()

lmbda_loss = np.zeros(((len(lmbdas), num_channels, len(layer_IDs))))
size_of_chunk = int(len(mapping_set) / num_folds)

r = 0
print(f"For fold={r}: ")
# get the sent ids for train and validation folds...
if r<(num_folds-1):
    val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
else:
    val_set = mapping_set[r*size_of_chunk:]
train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

# load features and spikes using the sent ids.
train_x = obj.unroll_features(
    bin_width=bin_width, sents=train_set, numpy=use_cpu, layer_IDs=layer_IDs
    )
train_y = obj.unroll_spikes(session, bin_width=bin_width, delay=delay,
                        sents=train_set, numpy=use_cpu)
train_x = cp.asnumpy(train_x).astype(np.float32)
train_y = cp.asnumpy(train_y).astype(np.float32)

# val_x = obj.unroll_features(
#                 bin_width=bin_width, sents=val_set, numpy=use_cpu, layer_IDs=layer_IDs
#                 )
# val_y = obj.unroll_spikes(session, bin_width=bin_width, delay=delay,
#                         sents=val_set, numpy=use_cpu)

# val_x = cp.asnumpy(val_x).astype(np.float32)
# val_y = cp.asnumpy(val_y).astype(np.float32)


# # computing Betas for lmbdas and removing train_x, train_y to manage memory..
# Betas = {}
# for i, lmbda in enumerate(lmbdas):
#     Betas[i] = utils.reg(train_x, train_y, lmbda)

start_time = time.time()
l = 0
ch = 32
x = train_x[l]
y = train_y[:, ch]


# model_coefficients, predicted_linear_response, is_converged, iter = tfp.glm.fit(
#                 model_matrix=x,
#                 response=y,
#                 model=tfp.glm.Poisson(),
#                 l2_regularizer=0.01,
#             )

end_time = time.time()
print(f"Time taken: {end_time - start_time} sec")