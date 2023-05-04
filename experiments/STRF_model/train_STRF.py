import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV, ElasticNetCV

import naplib as nl
from naplib.visualization import imSTRF

import auditory_cortex.helpers as helpers
import auditory_cortex.analysis.config as config
import auditory_cortex.analysis.analysis as analysis

import os
import time

start_time = time.time()
print("Starting out...")

# creating object of my STRF class..
strf = analysis.STRF()


current_time = time.time()
elapsed_time = current_time - start_time
print(f"Model created in {elapsed_time:.3f}")
print("Training the model...")

# choose an estimator...
ridge = False

# estimator = Ridge(alpha=1.0, max_iter=10)
# estimator = ElasticNet(l1_ratio=0.01)
# estimator=None
if ridge:
    alphas = np.logspace(-2,5, 8)
    estimator = RidgeCV(alphas=alphas, cv=5)
    filename = 'STRF_corr_RidgeCV'
else:
    # alphas = np.logspace(-2,5, 8)
    estimator = ElasticNetCV()
    filename = 'STRF_corr_elasticNetCV'

strf_model, corr = strf.fit(estimator, num_workers=4)


current_time = time.time()
elapsed_time = current_time - start_time
print(f"Model trained, it took {elapsed_time:.3f}")
print(corr.shape)
print(f"Correlation coefficient for RidgeCV model:", corr)

# saving correlation results...
# results = {'200206': corr}
path = os.path.join(config.results_dir, config.corr_sub_dir, filename)
np.save(path, corr)#results, allow_pickle=True)