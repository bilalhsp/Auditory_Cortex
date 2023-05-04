"""Saves configurations for corr_analysis class"""

# saved results dir
results_dir = '/depot/jgmakin/data/auditory_cortex/saved_results/'

# correlations..
corr_sub_dir = 'cross_validated_correlations'
corr_data_filename = 'c_w2l_correlations.csv'

#PCA_kde_data
pca_kde_sub_dir = 'PCA_kde_data'
pca_kde_data_filename = 'kde_computed_densities.pkl'
pca_dist_modes_filename = 'modes_pc_space_distributions.csv'



# task optimzation
task_optimization_paths = {
    'model_checkpoints_dir': '/scratch/gilbreth/ahmedb/wav2letter/modified_w2l/',
    'saved_corr_results_dir': '/depot/jgmakin/data/auditory_cortex/saved_results/task_optimization',

    }

# regression object configuration and saved checkpoint path..
regression_object_paths = {
    'model_param_path': '/home/ahmedb/projects/Wav2Letter/Wav2Letter/wav2letter/conf/config_rf.yaml',
    'saved_checkpoint': '/depot/jgmakin/data/auditory_cortex/pretrained_weights/w2l_modified/Wav2letter-epoch=024-val_loss=0.37.ckpt',
    'neural_data_dir': '/scratch/gilbreth/ahmedb/auditory_cortex'

}

# saved output path
svg_files = '/home/ahmedb/projects/Wav2Letter/saved_results/svg_files'


# # pretrained weights
# checkpoint_file = "Wav2letter-epoch=024-val_loss=0.37.ckpt"