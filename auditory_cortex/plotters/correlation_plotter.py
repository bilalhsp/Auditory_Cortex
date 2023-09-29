import matplotlib.pyplot as plt
from auditory_cortex import analysis


def session_bar_plot(
        session,
        **kwargs,
        ):
    """Bar plots for session correlations (mean across channels for all layers)"""
    
    separate = False
    if 'separate_color_maps' in kwargs: separate = kwargs.pop('separate_color_maps')
    if 'column' in kwargs: column = kwargs.pop('column')
    else: column = 'normalized_test_cc'
    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap') 
    else: 
        cmap='magma'
    if 'ax' in kwargs:
        ax = kwargs.pop('ax') 
    else: 
        _, ax = plt.subplots()
    
    corr_obj = analysis.Correlations('wave2letter_modified_opt_neural_delay')
    corr = corr_obj.get_session_corr(session)
    mean_layer_scores = corr.groupby('layer', as_index=False).mean()[column]
    num_layers = mean_layer_scores.shape[0]
    # print(mean_layer_scores.shape[0])
    if separate:
        vmin = mean_layer_scores.min()
        vmax = mean_layer_scores.max()
    else:
        vmin = 0
        vmax = 1

    plt.imshow(
        np.atleast_2d(mean_layer_scores), extent=(0,num_layers,0,4),
        cmap=cmap, vmin=vmin, vmax=vmax
    )