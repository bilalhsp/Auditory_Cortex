import yaml
from pathlib import Path

current_file = Path(__file__).resolve()
# Load the configuration file
config_file = current_file.parents[1] /'config.yml'
with open(config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

# setting the directories...
neural_data_dir = Path(config['neural_data_dir'])
pretrained_dir = Path(config['pretrained_models_dir'])
results_dir = Path(config['results_dir'])
cache_dir = Path(config['cache_dir'])

CACHE_DIR = cache_dir
normalizers_dir = cache_dir / 'normalizers'
saved_corr_dir = results_dir / 'cross_validated_correlations'
opt_inputs_dir = results_dir / 'optimal_inputs'
aux_dir = current_file.parents[0] / 'dnn_feature_extractor' /'auxilliary' 

# DNN and neural datasets
NEURAL_DATASETS = config['neural_datasets']
DNN_MODELS = config['dnn_models']
valid_model_names = DNN_MODELS

