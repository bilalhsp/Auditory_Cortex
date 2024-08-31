import os
import numpy as np
import scipy
import matplotlib.pyplot as plt


import auditory_cortex.utils as utils
from auditory_cortex import results_dir
from auditory_cortex.models import Regression
from auditory_cortex.analyses import Correlations
from auditory_cortex.dataloader import DataLoader
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex.plotters.coordinates_plotter import CoordinatesPlotter
from auditory_cortex.plotters.correlation_plotter import RegPlotter
from auditory_cortex.io_utils.io import read_WER


# ------------------  WER vs Neural prectibility ----------------#

# # # peak across all bin widths,
# trained_median_peaks_50ms = {
#     # peak correlation at bin width=50ms
#     'wav2letter_modified': 0.527,  # layer-ID: 4
#     'wav2vec2': 0.597,             # layer-ID: 9
#     'speech2text': 0.597,           # layer-ID: 8
#     'whisper_tiny': 0.611,          # layer-ID: 2
#     'whisper_base': 0.625,          # layer-ID: 2
#     'deepspeech2': 0.588,            # layer-ID: 2  
# }
# # Using different inclusion criteria....Need to automate thiss....
# trained_median_peaks_50ms = {
#     # peak correlation at bin width=50ms
#     'wav2letter_modified': 0.480,  # layer-ID: 4
#     'wav2vec2': 0.571,             # layer-ID: 9
#     'speech2text': 0.560,           # layer-ID: 8
#     'whisper_tiny': 0.597,          # layer-ID: 2
#     'whisper_base': 0.589,          # layer-ID: 2
#     'deepspeech2': 0.529,            # layer-ID: 2 
# }
# #     # 'wav2letter_modified': 0.500,  # peak at 60 ms, layer-ID: 4
# #     # 'wav2vec2': 0.584,             # peak at 40 ms, layer-ID: 8
# #     # 'speech2text': 0.583,           # peak at 60 ms, layer-ID: 5
# #     # 'whisper_tiny': 0.598,          # peak at 60 ms, layer-ID: 2
# #     # 'whisper_base': 0.599,          # peak at 40 ms, layer-ID: 2
# #     # 'deepspeech2': 0.583,           # peak at 100 ms, layer-ID: 2   
# # }


# # read off of the line plots..
# trained_median_peaks_20ms = {
#     'wav2letter_modified': 0.503,  # layer-ID: 8
#     'wav2vec2': 0.597,             # layer-ID: 7
#     'speech2text': 0.574,           # layer-ID: 7
#     'whisper_tiny': 0.600,          # layer-ID: 2
#     'whisper_base': 0.610,          # layer-ID: 2
#     'deepspeech2': 0.551,            # layer-ID: 2   
# }
# untrained_median_peaks = {
#     'wav2letter_modified': 0.24,  # layer-ID: 0
#     'wav2vec2': 0.356,             # layer-ID: 18
#     'speech2text': 0.530,           # layer-ID: 10
#     'whisper_tiny': 0.506,          # layer-ID: 1
#     'whisper_base': 0.507,          # layer-ID: 2
#     'deepspeech2': 0.534,            # layer-ID: 2   
# }

# trained_median_peaks = {
#     '20ms': trained_median_peaks_20ms,
#     '50ms': trained_median_peaks_50ms,
# }

def scatter_WER_v_corr(
		save_tikz=True,
		benchmark=None,
		bin_width=50,
		normalized = True,
		mVocs=False,
		trained_identifier = 'trained_all_bins', 
		use_stat_inclusion = True,
		inclusion_p_threshold = 0.01,
		use_poisson_null=True
		):
	"""Scatter plot WER on the benchmark specified vs peak median correlation."""
	df = read_WER()
	axes = []
	if benchmark is None:
		benchmarks = df.columns
	else:
		benchmarks = [benchmark]

	# Getting the median peaks for all the models...
	area = 'all'
	delay = 0
	poisson_normalizer = True

	median_peaks = {}
	for model_name in PlotterUtils.model_names:
		corr_obj_trained = Correlations(model_name+'_'+trained_identifier)
		threshold= corr_obj_trained.get_normalizer_threshold(
			bin_width=bin_width, poisson_normalizer=poisson_normalizer,
			mVocs=mVocs,
		)
		data_dist_trained = corr_obj_trained.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width, delay=delay,
				threshold=threshold, normalized=normalized,mVocs=mVocs,
				use_stat_inclusion=use_stat_inclusion,
				inclusion_p_threshold=inclusion_p_threshold,
				use_poisson_null=use_poisson_null,
			)

		layer_medians = [np.median(dist) for dist in data_dist_trained.values()]
		median_peaks[model_name] = np.max(layer_medians)
	
	for benchmark in benchmarks:
		fig, ax = plt.subplots()
		WERs = []
		rhos = []
		colors = []
		for model_name in PlotterUtils.model_names:
			WERs.append(df.loc[model_name, benchmark])
			rhos.append(median_peaks[model_name])
			colors.append(PlotterUtils.get_model_specific_color(model_name))
		
		ax.scatter(WERs, rhos, facecolors = colors)
		ax.set_title(f"{benchmark}")
		ax.set_xlabel("WER (%)")
		ax.set_ylabel("$\\rho$")
		axes.append(ax)
		# saving tikz file...
		if save_tikz:
			filepath = os.path.join(results_dir, 'tikz_plots', f"WER-vs-corr-{bin_width}ms-{benchmark}.tex")
			PlotterUtils.save_tikz(filepath)

	return axes





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
		save_tikz=True,
		normalized=True,
		threshold=None,
		mVocs=False,
		display_dotted_lines=False,
		display_inter_quartile_range=True,
		lag=None,
		model_identifier='STRF_freqs80_all_lags',
		ax=None,  
		use_stat_inclusion=False,
	):
	if ax is None:
		fig, ax = plt.subplots(figsize=(2,8))
	RegPlotter.plot_strf_baseline(
				area=area,
				bin_width=bin_width,
				delay=delay,
				alpha=alpha,
				lag=lag,
				ax=ax,
				save_tikz=save_tikz,
				display_dotted_lines=display_dotted_lines,
				display_inter_quartile_range=display_inter_quartile_range,
				normalized=normalized,
				threshold=threshold,
				mVocs=mVocs,
				model_identifier=model_identifier,
				use_stat_inclusion=use_stat_inclusion,
			)
	


# ------------------  Fig: peak layer core vs non-primary areas ----------------#


def peak_layer_core_non_primary_areas(
		model_name='whisper_base',
		bin_width=50,
		trained_identifier='trained_all_bins',
		baseline_identifier = f"STRF_freqs80_bw50",
		plot_baseline=False,
		untrained_identifiers=None,
		p_threshold = 0.01,
		offset_y=0.93,
		normalized=True,
		threshold=None,
		mVocs=False,
		save_tikz=True,
		use_stat_inclusion=True,
		inclusion_p_threshold=0.01,
		use_poisson_null=True,
	):
	"""Plots core vs non-primary areas plot across all layers and
	inidicates location of peak layer and other layers statistically similar
	to the peak layer"""
	areas = ['core', 'non-primary']
	area_wise_dist = {}
	for area in areas:
		dist_trained, *_ = RegPlotter.plot_all_layers_trained_and_shuffled(
			model_name=model_name,
			area=area,
			normalized=normalized,
			threshold=threshold,
			mVocs=mVocs,
			bin_width=bin_width,
			trained_identifier=trained_identifier,
			untrained_identifiers=untrained_identifiers,
			baseline_identifier=baseline_identifier,
			plot_baseline=plot_baseline,
			indicate_significance=False,
			save_tikz=False,
			use_stat_inclusion=use_stat_inclusion,
			inclusion_p_threshold=inclusion_p_threshold,
			use_poisson_null=use_poisson_null,
			)
		area_wise_dist[area] = dist_trained
		print(f"Number of neurons in dist for {area}: {dist_trained[4].size}")
		RegPlotter.indicate_peak_and_similar_layers(
			dist_trained, p_threshold=p_threshold, offset_y=offset_y)
		
		if save_tikz:
			filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-layerwise-with-peak-identified-{bin_width}ms-{model_name}-{area}.tex")
			PlotterUtils.save_tikz(filepath)
	return area_wise_dist


# ------------------  Fig: trained vs untrained networks plots ----------------#

def plot_trained_vs_shuffled_network_results(
		model_names = None,
		trained_identifier='trained_all_bins',
		untrained_identifiers=None,
		baseline_identifier=None,
		areas: list=None,
		normalized=True,
		threshold=None,
		mVocs=False,
		plot_difference=False,
		bin_width=20,
		alpha=0.1,
		save_tikz=True,
		sig_offset_x=0.0,
		sig_offset_y=0.92,
		sig_ind_size=8,
		arch_ind_offset=1,
		arch_ind_lw=8,
		keep_yticks=True,
		keep_xticks=True,
		plot_baseline=True,
		display_inter_quartile_range=True,
		display_dotted_lines=False,
		indicate_significance=True,
		tikz_indicator=None,
		use_stat_inclusion=False,
		inclusion_p_threshold=0.01,
		use_poisson_null=True,
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
	if model_names is None:
		model_names = PlotterUtils.model_names
	for model_name in model_names:
		for area in areas:
			dists = RegPlotter.plot_all_layers_trained_and_shuffled(
				model_name=model_name,
				bin_width=bin_width,
				area=area,
				normalized=normalized,
				threshold=threshold,
				mVocs=mVocs,
				alpha=alpha,
				save_tikz=save_tikz,
				sig_offset_x=sig_offset_x,
				sig_offset_y=sig_offset_y,
				sig_ind_size=sig_ind_size,
				arch_ind_offset=arch_ind_offset,
				arch_ind_lw=arch_ind_lw,
				keep_yticks=keep_yticks,
				keep_xticks=keep_xticks,
				plot_baseline=plot_baseline,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines,
				indicate_significance=indicate_significance,
				trained_identifier=trained_identifier,
				untrained_identifiers=untrained_identifiers,
				baseline_identifier=baseline_identifier,
				plot_difference=plot_difference,
				tikz_indicator=tikz_indicator,
				use_stat_inclusion=use_stat_inclusion,
				inclusion_p_threshold=inclusion_p_threshold,
				use_poisson_null=use_poisson_null,
				)


# ------------------  Fig: Trained networks (best layer) at each bin width ----------------#

def plot_best_layer_across_all_bin_widths(
	model_names: list=None,
	layer_ids: list=None,
	identifier = 'trained_all_bins',
	areas = None,
	normalized = True, 
	save_tikz = True, 
	alpha = 0.2, 
	normalizer_filename = None,
	display_inter_quartile_range=True,
	display_dotted_lines=False,
	norm_bin_width=None,
	p_threshold = 0.01,
	offset_y=0.93,
	):
	"""Plots and saves line plots for all networks, having correlation results
	of trained network (plotted in the network specific color), at all the 
	bin_widths analyzed i.e. [10 - 800] ms. 

	Args:
		untrained_identifier: str = identify shuffled/re-initialized, 
			choose from ['weights_shuffled', 'randn_weights', 'reset_weights']
		plot_difference: bool = plots dist of 'train - untrained' if True
		display_inter_quartile_range: bool = display shaded regions of dist
		plot_baseline: bool = 
		norm_bin_width: int = If spikes are predicted at a fixed bin width,
			and features are low pass filtered at different bin width (cut-off freq)
	"""
	if areas is None:
		areas = ['all'] #['core', 'belt']
	if model_names is None:
		model_names = PlotterUtils.model_names
	if layer_ids is None:
		layer_ids = [None for _ in model_names]
	else:
		assert len(layer_ids)==len(model_names), f"len(layer_ids)={len(layer_ids)} not equal to len(model_names)={len(model_names)}"
	for model_name, layer_id in zip(model_names, layer_ids):
		for area in areas:
			data_dist = RegPlotter.plot_best_layer_at_all_bin_width(
				model_name=model_name,
				area=area,
				alpha=alpha,
				identifier=identifier,
				save_tikz=save_tikz,
				normalized=normalized,
				normalizer_filename=normalizer_filename,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines,
				norm_bin_width=norm_bin_width,
				layer_id=layer_id,
				p_threshold=p_threshold,
				offset_y=offset_y,
			)
			# RegPlotter.indicate_peak_and_similar_layers(
			# data_dist, p_threshold=p_threshold, offset_y=offset_y)


# ------------------  Fig: Trained networks (best layer) at each bin width  ----------------#
# ------------------              using super set of tuned neurons          ----------------#

def plot_best_layer_across_all_bin_widths_using_super_set(
	areas = None, normalized = True, save_tikz = True, 
	alpha = 0.2, identifier = '_trained_all_bins',
	normalizer_filename = 'modified_bins_normalizer.csv',
	display_inter_quartile_range=True,
	display_dotted_lines=False,
	):
	"""Plots and saves line plots for all networks, having correlation results
	of trained network (plotted in the network specific color), at all the 
	bin_widths analyzed i.e. [10 - 800] ms. 

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
			data_dist = RegPlotter.plot_best_layer_at_all_bin_width_using_super_set(
				model_name=model_name,
				area=area,
				alpha=alpha,
				identifier=identifier,
				save_tikz=save_tikz,
				normalized=normalized,
				normalizer_filename=normalizer_filename,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines,
			)



# ------------------  Fig: spectrogram + spikes_counts + session coordinates ----------------#

def plot_spectrogram_spikes_counts_and_session_coordinates(
		model_names: list = None,
		sent_ids: list = None,
		sessions: list = None,
		chs: list = None, 
		bin_width=20,
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
		model_names = ['wav2vec2', 'whisper_base']
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
			bin_width=bin_width,
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
		bin_width = 20,
		trial_color=None,
		mean_color=None,
		prediction_color=None,
		saved_predictions=None,
		force_reload=False,
		save_tikz=True,
		plot_spectrogram=True,
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
		plot_spectrogram: bool = plots spectrogram if True.
	"""
		### best layers of the model..
	layers = {
		'wav2vec2': 8,
		'wav2letter_modified': 6,
		'deepspeech2': 3,
		'speech2text': 2,
		'whisper_tiny': 2,
		'whisper_base': 2
	}


	# plotting the spectrogram
	dataloader = DataLoader()

	assert sent_id in dataloader.metadata.test_sent_IDs, "Invalid sent ID, choose from"+\
					f" {list(dataloader.metadata.test_sent_IDs)}"
	# mdata = NeuralMetaData()
	
	if plot_spectrogram:
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
	# bin_width = 20
	sent_ids = [sent_id]
	mean_line_width = 2
	ind_trial_width = 1
	alpha = 0.3

	# # getting all trials...
	all_trials_spike_counts = dataloader.get_neural_data_for_repeated_trials(
			session, bin_width=bin_width,
			stim_ids=sent_ids
		)

	### getting predictions using the model..
	if saved_predictions is None:
		saved_predictions = {}

	prediction_colors = {}
	if model_name not in saved_predictions.keys():
		layer = layers[model_name]
		reg_obj = Regression(model_name)
		saved_predictions[model_name] = reg_obj.neural_prediction(
			session, bin_width=bin_width, delay=0, sents=sent_ids, layer_IDs=[layer],
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
	xtick_labels = xticks*bin_width
	ax.set_xticks(xticks, xtick_labels)

	ax.set_xlabel("time (ms)")
	ax.set_ylabel("spike count")
	return ax

# ------------------  Fig: Correlation gain by training ----------------#

def get_peak_dist_diff_trained_and_untrained(
		model_name,
		trained_identifier,
		untrained_identifiers,
		area='all',
		bin_width=50,
		delay=0,
		normalized=True,
		column=None,
		mVocs=False,
		poisson_normalizer=True,
		use_stat_inclusion=False,
		inclusion_p_threshold = 0.01,
		use_poisson_null=True,
	):
	"""Returns the difference of peak dist. in trained 
	and avg. of peak dist. for the untrained identifiers.
	"""
	corr_obj_trained = Correlations(model_name+'_'+trained_identifier)
	threshold= corr_obj_trained.get_normalizer_threshold(
		bin_width=bin_width, poisson_normalizer=poisson_normalizer,
		mVocs=mVocs,
	)
	data_dist_trained = corr_obj_trained.get_corr_all_layers_for_bin_width(
			neural_area=area, bin_width=bin_width, delay=delay,
			threshold=threshold, normalized=normalized, mVocs=mVocs,
			column=column, use_stat_inclusion=use_stat_inclusion,
			inclusion_p_threshold=inclusion_p_threshold,
			use_poisson_null=use_poisson_null,
			)


	data_dist_shuffled_list = []
	data_dist_shuffled = {}
	for untrained_identifier in untrained_identifiers:
		corr_obj_shuffled = Correlations(model_name+'_'+untrained_identifier)
		data_dist_shuffled_list.append(corr_obj_shuffled.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width, delay=delay,
				threshold=threshold, normalized=normalized, mVocs=mVocs,
				column=column, use_stat_inclusion=use_stat_inclusion,
				inclusion_p_threshold=inclusion_p_threshold,
				use_poisson_null=use_poisson_null,
			))
	for key in data_dist_shuffled_list[0]:
		distributions = list([dist[key] for dist in data_dist_shuffled_list])
		data_dist_shuffled[key] = np.stack(distributions, axis=0)
		data_dist_shuffled[key] = np.mean(data_dist_shuffled[key], axis=0)
	
	# peak_dist_trained = PlotterUtils.get_dist_with_peak_median(data_dist_trained)   
	# peak_dist_shuffled = PlotterUtils.get_dist_with_peak_median(data_dist_shuffled)
	# dist_diff = peak_dist_trained - peak_dist_shuffled

	# using max corr (across all layer) for each neuron
	all_layers_trained = np.stack(list(data_dist_trained.values()))
	peak_dist_trained = np.max(all_layers_trained, axis=0)
	all_layers_shuffled = np.stack(list(data_dist_shuffled.values()))
	peak_dist_shuffled = np.max(all_layers_shuffled, axis=0)
	dist_diff = peak_dist_trained - peak_dist_shuffled
	return dist_diff

def plot_diff_of_peak_dist_core_vs_non_primary(
		trained_identifier,
		untrained_identifiers,
		model_names=None,
		bin_width=50,
		normalized=True,
		mVocs=False,
		save_tikz=False,
		p_threshold=0.05,
		size=15,
		height = 0.6,
		
	):
	if model_names is None:
		model_names = PlotterUtils.model_names

	for model_name in model_names:
		areas = ['core', 'non-primary']
		distributions = {}
		for area in areas:

			diff = get_peak_dist_diff_trained_and_untrained(
				model_name,
				trained_identifier,
				untrained_identifiers,
				area=area,
				bin_width=bin_width,
				normalized=normalized,
				mVocs=mVocs,
				)
			distributions[area] = diff

		color = PlotterUtils.get_model_specific_color(model_name)
		ax=PlotterUtils.plot_box_whisker_swarm_plot(
			distributions,
			color=color,
		)
		ax.set_title(model_name)
		ax.set_ylim([-0.3, 0.7])
		# pvalue = scipy.stats.ttest_ind(
		# 	distributions['non-primary'], distributions['core'], 
		# 	equal_var=False, alternative='greater'
		# 	).pvalue
		pvalue = scipy.stats.mannwhitneyu(
			distributions['non-primary'], distributions['core'], 
			alternative='greater',
			).pvalue
		print(f"For {model_name}, p-value: {pvalue}")
		# significance condition..
		if pvalue < p_threshold:
			ax.scatter([0.5], [height+0.035], color='k', marker='*', s=size)
			ax.plot([0, 1.0], [height,height], color='k',)

		if save_tikz:
			filepath = os.path.join(
				results_dir, 'tikz_plots',
				f"training-gain-core-v-others-{bin_width}ms-{model_name}.tex"
				)
			PlotterUtils.save_tikz(filepath)







# ------------------  Supp. Fig: Wav2letter_spect corr plot for RFs and num_units ----------------#
def plot_layerwise_correlations_at_num_units_and_rfs(
		bin_width = 50,
		rfs :list = None,
		num_units :list = None,
		area='all',
		normalized=True,
		column=None,
		alpha=0.2,
		color=None,
		display_inter_quartile_range=True,
		display_dotted_lines=False,
		save_tikz=True,
		

		use_stat_inclusion=False,
		inclusion_p_threshold=0.05,
		use_poisson_null=True,
	):
	"""Plots correlation plots for all layers of the 'wav2letter_spect', 
	for supplemental figure. Make all such plots for different architectures
	desgined for RFs and number of units..
	"""
	model_name='wav2letter_spect'
	
	if num_units is None:
		num_units = [256, 512, 1024, 2048]
	if rfs is None:
		rfs = [65, 145, 225, 785]
	if color is None:
		color = PlotterUtils.colors[6]

	for rf in rfs:
		for units in num_units:
			identifier = f'units{units}_rf{rf}'
			dist, ax = RegPlotter.plot_all_layers_at_bin_width(
				model_name=model_name, identifier=identifier,
				color=color,
				area=area,
				normalized=normalized,
				column=column,
				alpha=alpha,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines,
				use_stat_inclusion=use_stat_inclusion,
				inclusion_p_threshold=inclusion_p_threshold,
				use_poisson_null=use_poisson_null,

			)
			ax.set_title(f'units{units}_rf{rf}')
			if save_tikz:
				filepath = os.path.join(
					results_dir,
					'tikz_plots',
					f"{model_name}-bw{bin_width}ms-rf{rf}-units{units}.tex",
				)
				PlotterUtils.save_tikz(filepath)

