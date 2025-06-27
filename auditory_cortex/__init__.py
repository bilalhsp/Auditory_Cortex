# standard libraries
import os
import yaml
import numpy as np
# from pycolormap_2d import ColorMap2DBremm, ColorMap2DZiegler

# select 2d color map to be used


# get parent directory of directory containing __init__.py file (parent of parent)
# aux_dir = os.path.join(os.path.dirname(__file__), 'computational_models', 'auxilliary')
aux_dir = os.path.join(os.path.dirname(__file__), 'dnn_feature_extractor', 'auxilliary')


# experiment configuration directory...!
config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
with open(os.path.join(config_dir, 'regression_config.yaml'), 'r') as f:
    config = yaml.load(f, yaml.FullLoader)


NEURAL_DATASETS = config['neural_datasets']
DNN_MODELS = config['dnn_models']
valid_model_names = DNN_MODELS

hpc_cluster = config['hpc_cluster']

# LPF_analysis_bw = config['LPF_analysis_bw']

# setting the directories...
neural_data_dir = config['neural_data_dir']
results_dir = config['results_dir']
saved_corr_dir = os.path.join(results_dir, 'cross_validated_correlations')
pretrained_dir = os.path.join(results_dir, 'pretrained_weights')
opt_inputs_dir = os.path.join(results_dir, 'optimal_inputs')

# Used to cache frequently used data (e.g. features, spikes etc.)
cache_dir = config['cache_dir']
CACHE_DIR = config['cache_dir']
normalizers_dir = os.path.join(cache_dir, 'normalizers')

