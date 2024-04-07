import os
import numpy as np
import matplotlib.pyplot as plt


import auditory_cortex.utils as utils
from auditory_cortex import results_dir
from auditory_cortex.models import Regression
from auditory_cortex.dataloader import DataLoader
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex.plotters.coordinates_plotter import CoordinatesPlotter
from auditory_cortex.plotters.correlation_plotter import RegPlotter



# ------------------  saving regression results as pickle (for Makin) ----------------#


def save_regression_correlations_all_networks():
    """Saving regression correlations (at 20 ms, normalized, all networks)
    to share the pickle file with makin.
    """
    model_names = PlotterUtils.model_names
    for model_name in model_names:
        RegPlotter.save_regression_correlations_for_model(model_name)


# ------------------  Fig: STRF baseline plot       ----------------#

def plot_strf_baseline(            
        area='all', bin_width=20, delay=0, alpha=0.1,
        save_tikz=True, normalized=True,
        display_dotted_lines=False,
        display_inter_quartile_range=True,   
    ):

    fig, ax = plt.subplots(figsize=(2,8))
    RegPlotter.plot_strf_baseline(
                area=area,
                bin_width=bin_width,
                delay=delay,
                alpha=alpha,
                ax=ax,
                save_tikz=save_tikz,
                display_dotted_lines=display_dotted_lines,
                display_inter_quartile_range=display_inter_quartile_range,
                normalized=normalized

            )



# ------------------  Fig: trained vs untrained networks plots ----------------#

def plot_trained_vs_shuffled_network_results(
        trained_identifier='bins_corrected_100',
        untrained_identifier='weights_shuffled',
        areas: list=None,
        plot_difference=False,
        bin_width=20,
        alpha=0.1,
        save_tikz=True,
        sig_offset_x=0.0,
        sig_offset_y=0.92,
        arch_ind_offset=1,
        arch_ind_lw=8,
        keep_yticks=True,
        keep_xticks=True,
        plot_baseline=False,
        display_inter_quartile_range=True,
        display_dotted_lines=False
    ):
    """Plots and saves line plots for all networks, having correlation results
    of trained network (plotted in the network specific color) and for network with 
    shuffled/re-initialized weights (plotted in black color). 

    Args:
        untrained_identifier: str = identify shuffled/re-initialized, 
            choose from ['weights_shuffled', 'randn_weights', 'reset_weights']
        plot_difference: bool = plots dist of 'train - untrained' if True
        display_inter_quartile_range: bool = display shaded regions of dist
        plot_baseline: bool = 
    """
    if areas is None:
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
                sig_offset_x=sig_offset_x,
                sig_offset_y=sig_offset_y,
                arch_ind_offset=arch_ind_offset,
                arch_ind_lw=arch_ind_lw,
                keep_yticks=keep_yticks,
                keep_xticks=keep_xticks,
                plot_baseline=plot_baseline,
                display_inter_quartile_range=display_inter_quartile_range,
                display_dotted_lines=display_dotted_lines,
                trained_identifier=trained_identifier,
                untrained_identifier=untrained_identifier,
                plot_difference=plot_difference
                )

    


# ------------------  Fig: spectrogram + spikes_counts + session coordinates ----------------#

def plot_spectrogram_spikes_counts_and_session_coordinates(
        model_names: list = None,
        sent_ids: list = None,
        sessions: list = None,
        chs: list = None, 
        trial_color=None,
        mean_color=None,
        prediction_color=None,
        saved_predictions=None,
        force_reload=False,
        save_tikz=True
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
        model_names = ['wave2vec2', 'whisper_base']
    if sent_ids is None:
        sent_ids = [12, 32] # blanket and moistioned lips
    if sessions is None:
        sessions = [200206, 180731]     #180807, ch-1
    if chs is None:
        chs = [32, 7]

    highlight_sessions = {}
    for i, model_name in enumerate(model_names):
        # saving spectograms and spikes counts (both actual & predicted) 
        plot_spectrogram_spike_count_pair(
            model_name=model_name, 
            sent_id=sent_ids[i],
            session=sessions[i],
            ch=chs[i],
            trial_color=trial_color,
            mean_color=mean_color,
            prediction_color=prediction_color,
            saved_predictions=saved_predictions,
            force_reload=force_reload,
            save_tikz=save_tikz
        )

        highlight_sessions[sessions[i]] = PlotterUtils.get_model_specific_color(model_name)

    # saving coordinate plots...
    coor_obj = CoordinatesPlotter()
    subjects = ['c_LH', 'c_RH', 'b_RH', 'f_RH']

    for subject in subjects:
        ax = coor_obj.scatter_sessions_for_recording_config(
            subject,
            save_tikz=save_tikz,
            highlight_sessions=highlight_sessions,
            )





def plot_spectrogram_spike_count_pair(
        model_name,
        sent_id,
        session,
        ch, 
        trial_color=None,
        mean_color=None,
        prediction_color=None,
        saved_predictions=None,
        force_reload=False,
        save_tikz=True
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
        ### best layers of the model..
    layers = {
        'wave2vec2': 7,
        'wave2letter_modified': 6,
        'deepspeech2': 3,
        'speech2text': 2,
        'whisper_tiny': 3,
        'whisper_base': 3
    }


    # plotting the spectrogram
    dataloader = DataLoader()

    assert sent_id in dataloader.metadata.test_sent_IDs, "Invalid sent ID, choose from"+\
                    f" {list(dataloader.metadata.test_sent_IDs)}"
    # mdata = NeuralMetaData()
    aud = dataloader.metadata.stim_audio(sent=sent_id)
    spect = utils.SyntheticInputUtils.get_spectrogram(aud).cpu().numpy().transpose()

    fig, ax = plt.subplots()
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
    all_trials_spike_counts = dataloader.get_neural_data_for_repeated_trials(
            session, bin_width=bin_width,
            sent_IDs=sent_ids
        )

    ### getting predictions using the model..
    if saved_predictions is None:
        saved_predictions = {}

    prediction_colors = {}
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
        filepath = os.path.join(
            results_dir, 'tikz_plots',
            f"neural-activity-sent{sent_id}-session-{session}-ch{ch}-{bin_width}ms-pred-using-{model_name}.tex"
            )
        PlotterUtils.save_tikz(filepath)

    return saved_predictions, all_trials_spike_counts



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
