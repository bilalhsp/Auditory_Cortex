import os
import numpy as np
import matplotlib.pyplot as plt


import auditory_cortex.utils as utils
from auditory_cortex import results_dir
from auditory_cortex.models import Regression
from auditory_cortex.dataloader import DataLoader
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex.plotters.correlation_plotter import RegPlotter



# ------------------  saving regression results as pickle (for Makin) ----------------#


def save_regression_correlations_all_networks():
    """Saving regression correlations (at 20 ms, normalized, all networks)
    to share the pickle file with makin.
    """
    model_names = PlotterUtils.model_names
    for model_name in model_names:
        RegPlotter.save_regression_correlations_for_model(model_name)

    




def plot_trained_vs_shuffled_network_results(
        bin_width=20,
        alpha=0.1,
        save_tikz=True,
        pos_sig_ind=0.93,
        keep_yticks=True,
        keep_xticks=True,
        plot_baseline=True,
        display_inter_quartile_range=False,
        untrained_identifier='_weights_shuffled',
    ):
    """Plots and saves line plots for all networks, specified neural area.
    """
    areas = ['all'] #['core', 'belt']
    model_names = PlotterUtils.model_names
    for model_name in model_names:
        for area in areas:
            dists = RegPlotter.plot_all_layers_trained_and_shuffled(
                model_name=model_name,
                bin_width=bin_width,
                area=area,
                alpha=alpha,
                save_tikz=save_tikz,
                pos_sig_ind=pos_sig_ind,
                keep_yticks=keep_yticks,
                keep_xticks=keep_xticks,
                plot_baseline=plot_baseline,
                display_inter_quartile_range=display_inter_quartile_range,
                untrained_identifier=untrained_identifier
                )

    




def plot_spike_counts(
        bin_width,
        ch,
        all_trials,
        predicted_spike_count,
        trial_color='gray',
        mean_color='k',
        prediction_colors=None,
        ind_trial_width=1,
        mean_line_width=2,
        alpha=0.3,
        xtick_label_step_ms=400,
        ax=None
    ):
    """Plots spike couts for individual trials, mean of all trials and prediction outcome"""
    if ax is None:
        fig, ax = plt.subplots()

    # plotting individual trials...
    for tr in range(all_trials.shape[0]):
        ax.plot(all_trials[tr,:,ch], color=trial_color, linewidth=ind_trial_width,
                alpha=alpha)
        
    # plotting mean...
    mean = np.mean(all_trials[...,ch], axis=0)
    ax.plot(mean, color=mean_color, linewidth=mean_line_width)

    # plotting predicted spike count...
    for model_name, predictions in predicted_spike_count.items():
        if prediction_colors is None:
            color = 'red'
            ax.plot(predictions[:,ch].squeeze(), color=color)
        else:
            try:
                color = prediction_colors[model_name]
                ax.plot(predictions[:,ch].squeeze(), color=color)
            except:
                continue
    
        

    # setting xlim..
    total_bins = all_trials.shape[1]
    ax.set_xlim([0, total_bins-1])

    # setting xtick labels...
    xtick_label_step_samples = int(xtick_label_step_ms/bin_width)
    xticks = np.arange(0, total_bins, xtick_label_step_samples)
    xtick_labels = xticks*xtick_label_step_samples
    ax.set_xticks(xticks, xtick_labels)

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("spike count")
    return ax






def plot_spectrogram_spike_count_pair(
    model_names: list=None, sent_id=12,
    session=200206, ch=62, 
    trial_color=None, mean_color=None,
    prediction_color=None,
    saved_predictions=None,
    force_reload=False, save_tikz=True
    ):
    """Plots spectogram for audio stimulus and spike counts (both actual and predicted)

    Args:
        model_name: model used for prediction
        sent_id: ID of the sent used for plot..be sure to choose from 
            the list of sent IDs having repeated trials, 
            i.e. [12, 13, 32, 43, 56, 163, 212, 218, 287, 308]
        session: ID of session to be used for spike count
        ch: ID of channel (from chosen session)
        trial_color: color used to display individual trials..
        mean_color: color used to display mean trials.
        prediction_color: color used to display predicted spike count.
        saved_predictions: input the dictionary of saved predictions..
        force_reload: bool = force reloading features for NN.
    """
    if model_names is None:
        model_names = ['wave2letter_modified']


    # plotting the spectrogram
    dataloader = DataLoader()

    assert sent_id in dataloader.metadata.test_sent_IDs, "Invalid sent ID, choose from"+\
                    f" {list(dataloader.metadata.test_sent_IDs)}"
    # mdata = NeuralMetaData()
    aud = dataloader.metadata.stim_audio(sent=sent_id)
    spect = utils.SyntheticInputUtils.get_spectrogram(aud).cpu().numpy().transpose()

    PlotterUtils.plot_spectrogram(
            spect,
            # cmap='viridis'
        )

    if save_tikz:
        filepath = os.path.join(results_dir, 'tikz_plots', f"spectrogram_sent{sent_id}.tex")
        PlotterUtils.save_tikz(filepath)

    # plotting individual trials, mean and predicted spike count...
    session = str(session)
    bin_width = 20
    sent_ids = [sent_id]
    mean_line_width = 2
    ind_trial_width = 1
    alpha = 0.3

    # # getting all trials...
    # neural_data = NeuralData(session)
    # all_trials_spike_counts = neural_data.get_repeated_trials(sents=sent_ids, bin_width=bin_width)

    all_trials_spike_counts = dataloader.get_neural_data_for_repeated_trials(
            session, bin_width=bin_width,
            sent_IDs=sent_ids
        )

    ### best layers of the model..
    layers = {
        'wave2vec2': 7,
        'wave2letter_modified': 6,
        'deepspeech2': 3,
        'speech2text': 2,
        'whisper_tiny': 3,
        'whisper_base': 3
    }

    ### getting predictions using the model..
    if saved_predictions is None:
        saved_predictions = {}

    prediction_colors = {}
    for model_name in model_names:
        if model_name not in saved_predictions.keys():
            layer = layers[model_name]
            reg_obj = Regression(model_name)
            saved_predictions[model_name] = reg_obj.neural_prediction(
                session, bin_width=bin_width, sents=sent_ids, layer_IDs=[layer],
                force_reload=force_reload
                )
        if prediction_color is None:
            prediction_colors[model_name] = PlotterUtils.get_model_specific_color(model_name)
        else:
            prediction_colors[model_name] = prediction_color
        

    ### plotting spike count waveforms...

    if trial_color is None:
        trial_color = 'gray'
    if mean_color is None:
        mean_color = 'k'
    

    ax = plot_spike_counts(
        bin_width,
        ch, all_trials_spike_counts, saved_predictions,
        trial_color, mean_color, prediction_colors,
        ind_trial_width, mean_line_width, alpha=alpha
    )
    # setting range of x-axis...
    total_bins = all_trials_spike_counts.shape[1]
    ax.set_xlim([0, total_bins-1])

    if save_tikz:
        if len(model_names) > 1:
            model_suffix = str(len(model_names)) + '-models'
        else:
             model_suffix = model_names[0]
        filepath = os.path.join(
            results_dir, 'tikz_plots',
            f"neural-activity-sent{sent_id}-ch{ch}-{bin_width}ms-pred-using-{model_suffix}.tex"
            )
        PlotterUtils.save_tikz(filepath)

    return saved_predictions, all_trials_spike_counts
