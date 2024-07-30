import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib as mpl

# local imports
from auditory_cortex import results_dir
from auditory_cortex.analyses import Correlations, STRFCorrelations
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex.io_utils.io import read_reg_corr, write_reg_corr
# from auditory_cortex.neural_data import NeuralMetaData

class RegPlotter:

	@staticmethod
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
		
		corr_obj = Correlations('wav2letter_modified_opt_neural_delay')
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


	# def plot_line_with_shaded_region(data_dict, model_name, alpha=0.2, ax=None):
	#     if ax is None:
	#         fig, ax = plt.subplots()
	#     color = PlotterUtils.get_model_specific_color(model_name)

	@staticmethod
	def scatter_best_layer_depth(
			bin_width = 20,
			# threshold = 0.061,
			identifier = '_bins_corrected_100',
			normalized = True,
			save_tikz=True
		):
		"""Makes a scatter plot of relative depth of best layer 
		for 'belt' vs 'core'
		"""
		model_names = PlotterUtils.model_names
		for model_name in model_names:
			
			area_wise_peak_layers = {}
			area_wise_errors_bars = {}
			corr_obj = Correlations(model_name+identifier)

			threshold = corr_obj.get_normalizer_threshold(
				bin_width=bin_width, poisson_normalizer=True
				)
			for area in ['core', 'belt']:
				# print(f"Object created for {model_name}, with id {identifier}")
				corr_dict = corr_obj.get_corr_all_layers_for_bin_width(
					neural_area=area, bin_width=bin_width,
					delay=0, threshold=threshold,
					normalized=normalized
				)

				# ############################  Picking median..######
				# layer_medians = {np.median(v):k for k,v in corr_dict.items()}
				# peak_median = max(layer_medians)
				# peak_layer = layer_medians[peak_median]
				# total_layers = len(corr_dict)
				# area_wise_peak_layers[area] = [peak_layer/total_layers]

				# # error bars....
				# neg_error = peak_median - np.percentile(corr_dict[peak_layer], q=25)
				# pos_error = np.percentile(corr_dict[peak_layer], q=75) - peak_median

				# area_wise_errors_bars[area] = np.array([neg_error, pos_error])[:,None]

				############## Using distribution of peak layers (for all channels)
				all_layers_combined = np.concatenate(
						[layer_spread[:,None] for _, layer_spread in corr_dict.items()],
						axis=1
					)
				
				num_layers = all_layers_combined.shape[1]
				peaks_layers_dist = np.argmax(all_layers_combined, axis=1)/num_layers

				corr_weights = np.max(all_layers_combined, axis=1)
				# corr_weights = np.square(corr_weights)
				weighted_mean = np.sum(peaks_layers_dist*corr_weights)/np.sum(corr_weights)
				area_wise_peak_layers[area] = weighted_mean
				
				# area_wise_peak_layers[area] = np.median(peaks_layers_dist)
				# neg_error = area_wise_peak_layers[area] - np.percentile(peaks_layers_dist, q=25)
				# pos_error = np.percentile(peaks_layers_dist, q=75) - area_wise_peak_layers[area]
				# area_wise_errors_bars[area] = np.array([neg_error, pos_error])[:,None]

			
			color = PlotterUtils.get_model_specific_color(model_name=model_name)
			# plt.scatter(
			plt.errorbar(
				area_wise_peak_layers['core'],
				area_wise_peak_layers['belt'],
				# xerr = area_wise_errors_bars['core'], 
				# yerr = area_wise_errors_bars['belt'],
				fmt='o',
				color=color,
				label=f'{model_name}'
				)

		x = np.linspace(0,1,20)
		plt.plot(x, x, 'k--', label='y=x')
		plt.xlabel(f"Relative best layer in 'core'")
		plt.ylabel(f"Relative best layer in 'belt'")

		plt.legend()        
		if save_tikz:
			filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-best-layer-depth-core-vs-belt_{bin_width}ms.tex")
			PlotterUtils.save_tikz(filepath)





	
	@staticmethod
	def plot_line_with_shaded_region(data_dict, color, alpha=0.2,
			shaded_low_percentile=25, shaded_high_percentile=75,
			dotted_low_percentile=5, dotted_high_percentile=95,
			ax=None,
			display_inter_quartile_range=True,
			display_dotted_lines=True,
			median_ls='-',
			
		):

		if ax is None:
			fig, ax = plt.subplots()
		
		medians = []
		x_coordinates = []
		shaded_lower_percentiles = []
		shaded_higher_percentiles = []
		dotted_lower_percentiles = []
		dotted_higher_percentiles = []
		
		peak_median_corr = 0
		peak_median_layer = 0

		maxs = []

		peak_max_corr = 0
		peak_max_layer = 0

		for layer_ID, layer_data in data_dict.items():
			# print(f"Layer-{layer_ID}: median: {np.median(layer_data)}, max: {np.max(layer_data)}")
			medians.append(np.median(layer_data))
			maxs.append(np.max(layer_data))
			x_coordinates.append(layer_ID)
			shaded_lower_percentiles.append(np.percentile(layer_data, shaded_low_percentile))
			shaded_higher_percentiles.append(np.percentile(layer_data, shaded_high_percentile))
			dotted_lower_percentiles.append(np.percentile(layer_data, dotted_low_percentile))
			dotted_higher_percentiles.append(np.percentile(layer_data, dotted_high_percentile))

			if medians[-1] > peak_median_corr:
				peak_median_corr = medians[-1]
				peak_median_layer = layer_ID

			if maxs[-1] > peak_max_corr:
				peak_max_corr = maxs[-1]
				peak_max_layer = layer_ID

		print(f"Peak corr (median): {peak_median_corr}, occurs at x_coordinate: {peak_median_layer}")
		print(f"Peak corr (max): {peak_max_corr}, occurs at x_coordinate: {peak_max_layer}")


		ax.plot(x_coordinates, medians, color=color, linestyle=median_ls)
		if display_inter_quartile_range:
			ax.fill_between(x=x_coordinates, y1=shaded_lower_percentiles, y2=shaded_higher_percentiles,
				alpha=alpha, color=color)
		# dotted lines...
		if display_dotted_lines:
			ax.plot(x_coordinates, dotted_lower_percentiles, '--', color=color)
			ax.plot(x_coordinates, dotted_higher_percentiles, '--', color=color)
		return ax

# ------------------  all layers at bin_width ----------------#

	@staticmethod
	def plot_all_network_layers_at_bin_width(model_name, area='core', bin_width=20,
				delay=0, alpha=0.1, save_tikz=True, normalized=True,
				identifier='_bins_corrected_100',
				pos_sig_ind = 0.95, p_threshold = 0.01,
				threshold=None,
				plot_baseline=False,
				keep_yticks = False,
				keep_xticks = False,
				indicate_significance=True,
				column=None,
				display_inter_quartile_range=False,
				display_dotted_lines=False,
		):
		
		corr_obj = Correlations(model_name+identifier)

		if threshold is None:
			threshold= corr_obj.get_normalizer_threshold(
				bin_width=bin_width, poisson_normalizer=True
			)
		corr_dist_all_layers = corr_obj.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width,
				delay=delay, threshold=threshold,
				normalized=normalized,
				column=column
			)

		color = PlotterUtils.get_model_specific_color(model_name)
		ax=RegPlotter.plot_line_with_shaded_region(
				data_dict=corr_dist_all_layers, color=color, alpha=alpha,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines
			)
		plt.title(f"{model_name}")
		plt.xlabel(f"Layer IDs")
		
		if column is None:
			plt.ylabel(f"$\\rho$")
			plt.ylim([0.0, 1.0])

		# get rid of the bounding boxes...
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)

		# xticks and yticks...
		if not keep_yticks:
			ax.set_yticks([])

		if not keep_xticks:
			ax.set_xticks([])

		# plot baseline...
		area_sessions = corr_obj.metadata.get_all_sessions(area)
		baseline_dist = corr_obj.get_baseline_corr_session(
			sessions= area_sessions,bin_width=bin_width, delay=delay,
					threshold=threshold, normalized=normalized)
		
		# statistical significance indicators
		# same baseline dist for every layer...
		dict_of_baseline_dist = {}
		for layer_ID in corr_dist_all_layers.keys():
			dict_of_baseline_dist[layer_ID] = baseline_dist

		if indicate_significance:
			RegPlotter.indicate_statistical_significance(
				corr_dist_all_layers,
				dict_of_baseline_dist,
				ax=ax, p = p_threshold, size=15,
				offset_y=pos_sig_ind
				)

		if plot_baseline:
			# same baseline dist for all layers...
			RegPlotter.plot_line_with_shaded_region(data_dict=dict_of_baseline_dist,
						color='gray', alpha=alpha, ax = ax,
						display_dotted_lines=False)

		if save_tikz:
			filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-layerwise-{area}-{model_name}.tex")
			PlotterUtils.save_tikz(filepath)

		return corr_dist_all_layers
	

	# def plot_strf_baseline(
	# 		area='all', bin_width=20, delay=0, alpha=0.1,
	# 		save_tikz=True, normalized=True,
	# 		ax=None, lag=None,
	# 		display_dotted_lines=False,
	# 		display_inter_quartile_range=True,
	# 		keep_xticks = True,
	# 		keep_yticks = True,
	# 		model_identifier='STRF_freqs80_all_lags', 
	# 		use_stat_inclusion=True,
	# 	):
	# 	# identifier = f'freqs80_100ms'
	# 	corr_obj = STRFCorrelations(model_identifier)
	# 	threshold= corr_obj.get_normalizer_threshold(
	# 		bin_width=bin_width, poisson_normalizer=True
	# 	)
	# 	baseline_dist_all_lags = {}
	# 	if lag is None:
	# 		lags = np.sort(corr_obj.data['tmax'].unique())    
	# 		xticks = np.arange(len(lags))        
	# 		for i, lag in enumerate(lags):
	# 			baseline_dist = corr_obj.get_correlations_for_bin_width( #get_corr_for_area
	# 				neural_area=area, bin_width=bin_width, delay=delay,
	# 				threshold=threshold, normalized=normalized, lag=lag,
	# 				use_stat_inclusion=use_stat_inclusion
	# 			)
	# 			key = xticks[i]
	# 			baseline_dist_all_lags[key] = baseline_dist
	# 			print(f"Number of samples in distribution: {baseline_dist.size}")           
	# 		lags_ms = lags *1000
	# 		xtick_labels = list(lags_ms.astype(int))
	# 		lag_ind = 'all-lags'
	# 		plt.xlabel(f"lags (ms)")
	# 	else:
	# 		xticks = [0,1]
	# 		xtick_labels = []
	# 		# get baseline results..        
	# 		baseline_dist = corr_obj.get_correlations_for_bin_width(
	# 			neural_area=area, bin_width=bin_width, delay=delay,
	# 			threshold=threshold, normalized=normalized, lag=lag,
	# 			use_stat_inclusion=use_stat_inclusion
	# 		)
	# 		print(f"Number of samples in distribution: {baseline_dist.size}")           
	# 		# repeating baseline distribution, to allow plotting horizontal line..
	# 		baseline_dist_all_lags = {key: baseline_dist.values for key in xticks}
	# 		xtick_labels = []
	# 		lag_ind = f'lag-{lag}'
	# 		plt.xlabel(f"   ")
	# 	# plotting the baseline..
	# 	model_name = 'STRF'
	# 	baseline_color = PlotterUtils.get_model_specific_color(model_name)
	# 	ax=RegPlotter.plot_line_with_shaded_region(data_dict=baseline_dist_all_lags,
	# 						color=baseline_color, alpha=alpha, ax=ax,
	# 						display_dotted_lines=display_dotted_lines,
	# 						display_inter_quartile_range=display_inter_quartile_range
	# 						)

	# 	plt.ylabel(f"$\\rho$")
	# 	plt.ylim([0.0, 1.0])
	# 	ax.set_xticks(xticks)
	# 	ax.set_xticklabels(xtick_labels)

	# 			## formatting the plot...
	# 	plt.title(f"{model_name}")
	# 			# get rid of the bounding boxes...
	# 	ax.spines['top'].set_visible(False)
	# 	ax.spines['bottom'].set_visible(False)
	# 	ax.spines['right'].set_visible(False)
	# 	ax.spines['left'].set_visible(False)

	# 	# xticks and yticks...
	# 	if not keep_yticks:
	# 		ax.set_yticks([])
	# 	if not keep_xticks:
	# 		ax.set_xticks([])

	# 	if save_tikz:
	# 		filepath = os.path.join(
	# 			results_dir,
	# 			'tikz_plots',
	# 			f"Reg-trained-{area}-bw{bin_width}ms-{lag_ind}-{model_name}.tex"
	# 			)
	# 		PlotterUtils.save_tikz(filepath)

	def plot_strf_baseline(
			area='all', bin_width=20, delay=0, alpha=0.1,
			save_tikz=True,
			normalized=True,
			mVocs=False,
			ax=None, lag=None,
			display_dotted_lines=False,
			display_inter_quartile_range=True,
			keep_xticks = True,
			keep_yticks = True,
			model_identifier='STRF_freqs80_all_lags', 
			use_stat_inclusion=True,
		):
		# identifier = f'freqs80_100ms'
		corr_obj = STRFCorrelations(model_identifier)
		threshold= corr_obj.get_normalizer_threshold(
			bin_width=bin_width, poisson_normalizer=True, mVocs=mVocs,
		)
		xticks = [0,1]
		xtick_labels = []
		baseline_dist = corr_obj.get_correlations_for_bin_width( #get_corr_for_area
					neural_area=area, bin_width=bin_width, delay=delay,
					threshold=threshold, normalized=normalized, mVocs=mVocs,
					lag=None, use_stat_inclusion=use_stat_inclusion
				)

		print(f"Number of samples in distribution: {baseline_dist.size}")           
		# repeating baseline distribution, to allow plotting horizontal line..
		baseline_dist_all_lags = {key: baseline_dist.values for key in xticks}
		xtick_labels = []
		# lag_ind = f'lag-{lag}'
		lag_ind = f'lag-cv'
		plt.xlabel(f"   ")
		# plotting the baseline..
		model_name = 'STRF'
		baseline_color = PlotterUtils.get_model_specific_color(model_name)
		ax=RegPlotter.plot_line_with_shaded_region(data_dict=baseline_dist_all_lags,
							color=baseline_color, alpha=alpha, ax=ax,
							display_dotted_lines=display_dotted_lines,
							display_inter_quartile_range=display_inter_quartile_range
							)

		plt.ylabel(f"$\\rho$")
		plt.ylim([0.0, 1.0])
		ax.set_xticks(xticks)
		ax.set_xticklabels(xtick_labels)

				## formatting the plot...
		plt.title(f"{model_name}")
				# get rid of the bounding boxes...
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)

		# xticks and yticks...
		if not keep_yticks:
			ax.set_yticks([])
		if not keep_xticks:
			ax.set_xticks([])

		if save_tikz:
			filepath = os.path.join(
				results_dir,
				'tikz_plots',
				f"Reg-trained-{area}-bw{bin_width}ms-{lag_ind}-{model_name}.tex"
				)
			PlotterUtils.save_tikz(filepath)



		# # baseline is the same for all networks
		# corr_obj = Correlations('deepspeech2'+'_'+identifier)
		# threshold= corr_obj.get_normalizer_threshold(
		#     bin_width=bin_width, poisson_normalizer=True
		# )

		# # get baseline results..        
		# baseline_dist = corr_obj.get_baseline_corr_for_area(
		#     neural_area=area, bin_width=bin_width, delay=delay,
		#     threshold=threshold, normalized=normalized
		# )
		# print(f"Number of samples in distribution: {baseline_dist.size}")           
		# # repeating baseline distribution, to allow plotting horizontal line..
		# x_ticks = [0,1]
		# baseline_dist_all_layer = {key: baseline_dist.values for key in x_ticks}

		# # plotting the baseline..
		# model_name = 'STRF'
		# baseline_color = PlotterUtils.get_model_specific_color(model_name)
		# ax=RegPlotter.plot_line_with_shaded_region(data_dict=baseline_dist_all_layer,
		#                     color=baseline_color, alpha=alpha, ax=ax,
		#                     display_dotted_lines=display_dotted_lines,
		#                     display_inter_quartile_range=display_inter_quartile_range
		#                     )
		
		# plt.ylabel(f"$\\rho$")
		# plt.ylim([0.0, 1.0])
		# ax.set_xticklabels([])

		#         ## formatting the plot...
		# plt.title(f"{model_name}")
		# plt.xlabel(f"   ")

		# # get rid of the bounding boxes...
		# ax.spines['top'].set_visible(False)
		# ax.spines['bottom'].set_visible(False)
		# ax.spines['right'].set_visible(False)
		# ax.spines['left'].set_visible(False)

		# # xticks and yticks...
		# if not keep_yticks:
		#     ax.set_yticks([])
		# if not keep_xticks:
		#     ax.set_xticks([])

		# if save_tikz:
		#     filepath = os.path.join(
		#         results_dir,
		#         'tikz_plots',
		#         f"Reg-trained-{area}-{model_name}.tex"
		#         )
		#     PlotterUtils.save_tikz(filepath)


	@staticmethod
	def plot_all_layers_trained_and_shuffled(
		model_name, area='all', bin_width=20, delay=0, alpha=0.1,
		save_tikz=True, normalized=True,
		# identifier='_bins_corrected_100',
		sig_offset_x=0,
		sig_offset_y=0.93,
		sig_ind_size=8,
		arch_ind_offset=1.0,
		arch_ind_lw=8,
		p_threshold=0.01,
		plot_baseline=False, 
		display_inter_quartile_range=True,
		display_dotted_lines=True,
		keep_xticks = True,
		keep_yticks = True,
		trained_identifier='trained_all_bins',
		untrained_identifiers=None,
		baseline_identifier=None,
		plot_difference=False,
		tikz_indicator=None,
		column=None,
		mVocs=False,
		indicate_significance=True,
		indicate_architecture=True,
		poisson_normalizer=True,
		use_stat_inclusion=False,
		inclusion_p_threshold = 0.01,
		use_poisson_null=True,
		):

		if untrained_identifiers is None:
			untrained_identifiers = [
				'reset_weights0', 'reset_weights1', 'reset_weights2', 'reset_weights3'
				]

		# select appropraite results identifier...
		if tikz_indicator is None:
			tikz_indicator = 'reset-avg'
			# if 'randn' in untrained_identifiers[0]:
			#         tikz_indicator = 'randn'
			# elif 'reset' in untrained_identifiers[0]:
			#         tikz_indicator = 'reset'
			# elif 'shuffle' in untrained_identifiers[0]:
			#         tikz_indicator = 'shuffled'
		
		# trained_network...
		# identifier='_bins_corrected_100'
		corr_obj_trained = Correlations(model_name+'_'+trained_identifier)
		threshold = corr_obj_trained.get_normalizer_threshold(
			bin_width=bin_width, poisson_normalizer=poisson_normalizer,
			mVocs=mVocs
		)
		data_dist_trained = corr_obj_trained.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width, delay=delay,
				threshold=threshold, normalized=normalized,
				column=column, mVocs=mVocs, use_stat_inclusion=use_stat_inclusion,
				inclusion_p_threshold=inclusion_p_threshold,
				use_poisson_null=use_poisson_null,

			)
		
		# weights shuffled ...
		# identifier='_weights_shuffled'
		data_dist_shuffled_list = []
		data_dist_shuffled = {}
		for untrained_identifier in untrained_identifiers:
			corr_obj_shuffled = Correlations(model_name+'_'+untrained_identifier)
			data_dist_shuffled_list.append(corr_obj_shuffled.get_corr_all_layers_for_bin_width(
					neural_area=area, bin_width=bin_width, delay=delay,
					threshold=threshold, normalized=normalized,
					column=column, mVocs=mVocs, use_stat_inclusion=use_stat_inclusion,
					inclusion_p_threshold=inclusion_p_threshold,
					use_poisson_null=use_poisson_null,
				))
		for key in data_dist_shuffled_list[0]:
			distributions = list([dist[key] for dist in data_dist_shuffled_list])
			data_dist_shuffled[key] = np.stack(distributions, axis=0)
			data_dist_shuffled[key] = np.mean(data_dist_shuffled[key], axis=0)
		
		# # get baseline results..        
		# baseline_dist = corr_obj_shuffled.get_baseline_corr_for_area(
		#     neural_area=area, bin_width=bin_width, delay=delay,
		#     threshold=threshold, normalized=normalized
		# )
		
		# plot trained network results...
		color = PlotterUtils.get_model_specific_color(model_name)
		if plot_difference:
			# get difference of trained and shuffled distributions..
			data_dist_diff = {}
			for (k, trained), (_, shuffled) in zip(data_dist_trained.items(), data_dist_shuffled.items()):

				diff = trained - shuffled
				data_dist_diff[k] = diff

			# plotting the diff distribution...
			ax=RegPlotter.plot_line_with_shaded_region(
				data_dict=data_dist_diff, color=color, alpha=alpha,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines,
				)
			
			if indicate_significance:
				# signigicance of 'difference' distribution...
				RegPlotter.indicate_statistical_significance(
					data_dist_diff,
					ax=ax, p = p_threshold,
					size=sig_ind_size,
					offset_y=sig_offset_y - 0.1,
					offset_x=sig_offset_x,
					color = 'k'
					)
			
			# horizontal line at y=0
			ax.axhline(y=0, xmin=0, xmax=len(data_dist_trained), color='k')

			plt.ylabel(f"$\\Delta \\rho$")
			plt.ylim([-0.4, 1.0])
			post_script = 'diff-'


		else:
			# plotting individual distributions...
			ax=RegPlotter.plot_line_with_shaded_region(
				data_dict=data_dist_trained, color=color, alpha=alpha,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines,
				)
			# plot shuffled network results...
			ax=RegPlotter.plot_line_with_shaded_region(
				data_dict=data_dist_shuffled, color='k', alpha=alpha, ax=ax,
				display_inter_quartile_range=display_inter_quartile_range,
				display_dotted_lines=display_dotted_lines,
				)
			
			if indicate_significance:
				# signigicance over 'shuffled' results...
				RegPlotter.indicate_statistical_significance(
					data_dist_trained,
					data_dist_shuffled,
					ax=ax, p = p_threshold,
					size=sig_ind_size,
					offset_y=sig_offset_y,
					offset_x=sig_offset_x,
					color = 'k'
					)
			
			# plot baseline...
			if plot_baseline:
				if baseline_identifier is None:
					baseline_identifier = 'STRF_freqs80_all_lags'
				strf_obj = STRFCorrelations(baseline_identifier)
                # Deprecated.
				# threshold= strf_obj.get_normalizer_threshold(
				# 	bin_width=bin_width, poisson_normalizer=True,
				# 	mVocs=mVocs
				# )
				baseline_dist = strf_obj.get_correlations_for_bin_width(
					neural_area=area, bin_width=bin_width, delay=delay,
					threshold=threshold, normalized=normalized, mVocs=mVocs,
					lag=None,   # saying lag=0.3
					use_stat_inclusion=use_stat_inclusion,
					inclusion_p_threshold=inclusion_p_threshold,
					use_poisson_null=use_poisson_null,
				)

				# baseline_dist = strf_obj.get_corr_for_area(
				#     neural_area=area, bin_width=bin_width, delay=delay,
				#     threshold=threshold, normalized=normalized, lag=None,   # saying lag=0.3
				# )

				# simply repeating baseline dist for all layers to allow for comparison....
				baseline_dist_all_layer = {key: baseline_dist.values for key in data_dist_trained.keys()}
				baseline_color = PlotterUtils.get_model_specific_color('baseline')
				# signigicance over 'baseline' results...
				if indicate_significance:
					RegPlotter.indicate_statistical_significance(
						data_dist_trained,
						baseline_dist_all_layer,
						ax=ax, p = p_threshold,
						size=sig_ind_size,
						offset_y=sig_offset_y - 0.1,
						offset_x=sig_offset_x,
						color = baseline_color
						)

			
				RegPlotter.plot_line_with_shaded_region(data_dict=baseline_dist_all_layer,
							color=baseline_color, alpha=alpha, ax = ax,
							display_dotted_lines=False,
							display_inter_quartile_range=False,
							median_ls='--'
							)
				
			post_script = ''

			plt.ylabel(f"$\\rho$")
			if column is None:
				plt.ylim([0.0, 1.0])

		if indicate_architecture:
			# indicate layer architecture type..
			architecture_specific_ids = corr_obj_trained.get_architecture_specific_layer_ids()
			RegPlotter.indicate_layer_architecture(
				ax, architecture_specific_ids, arch_ind_offset, arch_ind_lw
			)
		if mVocs:
			post_title = ', test: mVocs'
		else:
			post_title = ', test: timit'

		## formatting the plot...
		plt.title(f"{model_name}{post_title}")
		plt.xlabel(f"layer IDs")

		# get rid of the bounding boxes...
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)

		# xticks and yticks...
		if not keep_yticks:
			ax.set_yticks([])
		if not keep_xticks:
			ax.set_xticks([])

		if save_tikz:
			filepath = os.path.join(
				results_dir,
				'tikz_plots',
				f"Reg-trained-{post_script}{tikz_indicator}-{area}-bw{bin_width}ms-{model_name}.tex"
				)
			PlotterUtils.save_tikz(filepath)

		return data_dist_trained, data_dist_shuffled #, baseline_dist


	@staticmethod
	def plot_baseline_histogram_at_bin_width(
			area='core', bin_width=20,
			delay=0, threshold = 0.068, alpha=0.1, save_tikz=True, normalized=True,
			identifier='_bins_corrected_100',
			pos_sig_ind = 0.95, p_threshold = 0.01,
			plot_baseline=False,
			keep_yticks = True,
			keep_xticks = True

		):
		"""Histogram of baseline dist. corresponding to 
		'plot_all_network_layers_at_bin_width' for candidate networks.
		"""

		# any network could be used, since this is for the baseline
		model_name = 'wav2letter_modified'
		identifier = '_bins_corrected_100'
		corr_obj = Correlations(
				model_name+identifier,
		)
		bw_threshold = corr_obj.get_normalizer_threshold(
			bin_width=bin_width, poisson_normalizer=True
			)
		# plot baseline...
		area_sessions = corr_obj.metadata.get_all_sessions(area)
		baseline_dist = corr_obj.get_baseline_corr_session(
			sessions= area_sessions,bin_width=bin_width, delay=delay,
					threshold=bw_threshold, normalized=normalized)
		



# ------------------		corr plot for all layers of network		----------------#
	@staticmethod
	def plot_all_layers_at_bin_width(
			model_name='wav2letter_spect',
			identifier = 'units256_rf785',
			bin_width = 50,
			area='all',
			normalized=True,
			column=None,
			alpha=0.2,
			color=None,
			display_inter_quartile_range=True,
			display_dotted_lines=False,
			

			use_stat_inclusion=False,
			inclusion_p_threshold=0.05,
			use_poisson_null=True,
		):
		"""Plot correlation distributions for all the layers of the network."""
		corr_obj = Correlations(
		model_name=model_name+'_'+identifier,
		)

		threshold= corr_obj.get_normalizer_threshold(
			bin_width=bin_width, poisson_normalizer=True,
			)
		data_dist_trained = corr_obj.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width,
				threshold=threshold, normalized=normalized,
				column=column, use_stat_inclusion=use_stat_inclusion,
				inclusion_p_threshold=inclusion_p_threshold,
				use_poisson_null=use_poisson_null,
			)

		if color is None:
			color = PlotterUtils.get_model_specific_color(model_name)

		# plotting individual distributions...
		ax=RegPlotter.plot_line_with_shaded_region(
			data_dict=data_dist_trained, color=color, alpha=alpha,
			display_inter_quartile_range=display_inter_quartile_range,
			display_dotted_lines=display_dotted_lines,
			)

		## formatting the plot...
		plt.title(f"{model_name}")
		plt.xlabel(f"layer IDs")
		plt.ylabel(f"$\\rho$")
		if column is None:
			plt.ylim([0.0, 1.0])
		return data_dist_trained, ax

	



# ------------------  KDE: all layers at bin_width ----------------#

	@staticmethod
	def plot_KDE_all_layers_at_bin_width(
			model_name,
			area = 'core',
			bin_width = 20,
			delay = 0,
			cmap=None,
			adjust_color=True,
			ax=None,
			normalized = True,
			poisson_normalizer = True,
			identifier = '_bins_corrected_100',
			normalizer_filename = 'modified_bins_normalizer.csv'
		):

	
		corr_obj = Correlations(
				model_name+identifier,
				normalizer_filename=normalizer_filename
			)
		
		histograms, xticks, yticks = corr_obj.get_KDE_all_layers_for_bin_width(
			   neural_area=area, bin_width=bin_width
			)
		if cmap is None:
			
			cmap = PlotterUtils.get_model_specific_cmap(model_name)
			# color = PlotterUtils.get_model_specific_color(model_name)
			# if adjust_color:
			#     print(f"Adjusting peak color..")
			#     # change color to make it more darker...
			#     hsv_color = mpl.colors.rgb_to_hsv(color)
			#     # adjust value...
			#     hsv_color[1] = hsv_color[1] + 0.2
			#     hsv_color = np.clip(hsv_color, 0, 1)
			#     color = mpl.colors.hsv_to_rgb(hsv_color)

			# cmap = sns.light_palette(color, as_cmap=True)
		if ax is None:
			fig, ax = plt.subplots()

		ax.imshow(histograms, interpolation=None, origin='lower', cmap=cmap)
		# plt.colorbar()
		# adding spaces to xticks...
		if xticks[0].size <= 10:
			step_size = 2
		else:
			step_size = 5
		ax.set_xticks(xticks[0][::step_size], xticks[1][::step_size])
		ax.set_yticks(yticks[0], yticks[1])
		ax.set_xlabel(f"Layer IDs")
		ax.set_ylabel(f"$\\rho$")
		ax.set_title(f"{model_name}, all layers at bin_width: {bin_width}")

		filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-layerwise-heatmap-{area}-{model_name}.tex")
		PlotterUtils.save_tikz(filepath)

	@staticmethod
	def plot_KDE_for_baseline(
			area = 'core',
			bin_width = 20,
			delay = 0,
			cmap=None,
			ax=None,
			normalized = True,
			poisson_normalizer = True,
		):

		# any network could be used, since this is for the baseline
		model_name = 'wav2letter_modified'
		identifier = '_bins_corrected_100'
		corr_obj = Correlations(
				model_name+identifier,
			)
		
		histograms, xticks, yticks = corr_obj.get_KDE_for_baseline(
			   area=area, bin_width=bin_width, 
			   normalized=normalized, poisson_normalizer=poisson_normalizer
			)
		if cmap is None:
			
			cmap = PlotterUtils.get_model_specific_cmap('baseline')

		if ax is None:
			fig, ax = plt.subplots()

		ax.imshow(histograms, interpolation=None, origin='lower', cmap=cmap)
		# plt.colorbar()
		# ax.set_xticks(xticks[0], xticks[1])
		ax.set_xticks([])
		ax.set_yticks(yticks[0], yticks[1])
		ax.set_xlabel(f"Baseline")
		ax.set_ylabel(f"$\\rho$")
		ax.set_title(f"Baseline at bin_width: {bin_width}")

		filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-layerwise-heatmap-{area}-baseline.tex")
		PlotterUtils.save_tikz(filepath)


# ------------------  'Selected' layer at each bin_width ----------------#
	@staticmethod
	def plot_one_network_layer_at_all_bin_width(model_name, area='all', layer=6,
				delay=0, alpha=0.2, save_tikz=True, poisson_normalizer=True,
				identifier='_trained_all_bins', labels=True,
				normalized=True, normalizer_filename=None
		):
		
		corr_obj = Correlations(
			model_name+identifier,
			normalizer_filename=normalizer_filename
			)
		data_dist = corr_obj.get_corr_all_bin_widths_for_layer(
			neural_area=area, layer=layer,
			delay=delay, poisson_normalizer=poisson_normalizer,
			normalized=normalized
			)

		color = PlotterUtils.get_model_specific_color(model_name)
		ax=RegPlotter.plot_line_with_shaded_region(data_dict=data_dist,
					color=color, alpha=alpha)

		# RegPlotter.plot_line_with_shaded_region(data_dict=data_dist,
		#             model_name=model_name, alpha=alpha)
		if labels:
			plt.title(f"Regression: {model_name}, layer-{layer}, area-{area}")
			plt.xlabel(f"Bin widths")
			plt.ylabel(f"$\\rho$")
			plt.ylim([0.0, 1.0])

		if save_tikz:
			filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-all_bin_widths-layer{layer}-{area}-{model_name}.tex")
			PlotterUtils.save_tikz(filepath)

# ------------------  Best layer at each bin_width ----------------#
	@staticmethod
	def plot_best_layer_at_all_bin_width_using_super_set(
		model_name, area='all',
		normalized=True,
		poisson_normalizer=True,
		delay=0, alpha=0.2, save_tikz=True,
		identifier='_trained_all_bins',
		labels=True,
		normalizer_filename=None,
		display_inter_quartile_range=True,
		display_dotted_lines=False,
		):
		"""Plots best layers at each bin width for the model name specified,
		uses tuned superset of neurons i.e. Union of tuned neurons
		at all bin widths.
		"""
		
		corr_obj = Correlations(
			model_name+identifier,
			normalizer_filename=normalizer_filename
			)
		
		data_dist = {}
		sig_session_channel_dict = corr_obj.get_significant_session_and_channels_at_all_bin_width(
			poisson_normalizer=poisson_normalizer
		)
		bin_widths = np.sort(corr_obj.data['bin_width'].unique())
		for bin_width in bin_widths:

			data_dist[bin_width] = corr_obj.get_layer_dist_with_peak_median_using_super_set(
				sig_session_channel_dict, bin_width=bin_width, normalized=normalized,
				delay=delay,
			)
			# replace nan entries with zero..
			# dist[np.isnan(dist)] = 0
			# data_dist[bin_width] = dist

		color = PlotterUtils.get_model_specific_color(model_name)
		ax=RegPlotter.plot_line_with_shaded_region(
			data_dict=data_dist, color=color, alpha=alpha,
			display_inter_quartile_range=display_inter_quartile_range,
			display_dotted_lines=display_dotted_lines,
			)

		# RegPlotter.plot_line_with_shaded_region(data_dict=data_dist,
		#             model_name=model_name, alpha=alpha)
		if labels:
			plt.title(f"Regression: {model_name}, peak_median_layer, area-{area}")
			plt.xlabel(f"bin width (ms)")
			plt.ylabel(f"$\\rho$")
			plt.ylim([0.0, 1.0])

		if save_tikz:
			filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-all-bin-widths-best-layer-global-tuned-{area}-{model_name}.tex")
			PlotterUtils.save_tikz(filepath)
		return data_dist






	@staticmethod
	def plot_best_layer_at_all_bin_width(model_name, area='all',
				delay=0, alpha=0.2, save_tikz=True, poisson_normalizer=True,
				identifier='trained_all_bins', labels=True,
				normalized=True, normalizer_filename=None,
				display_inter_quartile_range=True,
				display_dotted_lines=False,
				norm_bin_width=None,
                layer_id=None,
				p_threshold = 0.01,
	            offset_y=0.93,
		):
		"""Plots best layers at each bin width for the model name specified,
		uses threshold method for selecting the 'tuned' neurons.
		
		Args:
            layer_id: int = If specified, returns dist for layer_id,
                    else returns the for layer with peak median.
		"""
		
		corr_obj = Correlations(
			model_name+'_'+identifier,
			normalizer_filename=normalizer_filename
			)
		
		data_dist = {}
		bin_widths = np.sort(corr_obj.data['bin_width'].unique())
		for bin_width in bin_widths:
			data_dist[bin_width] = corr_obj.get_layer_dist_with_peak_median(
				bin_width=bin_width, 
				neural_area=area, delay=delay,
				normalized=normalized, poisson_normalizer=poisson_normalizer,
				norm_bin_width=norm_bin_width, layer_id=layer_id
			)

		color = PlotterUtils.get_model_specific_color(model_name)
		ax=RegPlotter.plot_line_with_shaded_region(
			data_dict=data_dist, color=color, alpha=alpha,
			display_inter_quartile_range=display_inter_quartile_range,
			display_dotted_lines=display_dotted_lines,
			)
		
		RegPlotter.indicate_peak_and_similar_layers(
			data_dist, p_threshold=p_threshold, offset_y=offset_y)

		if labels:
			plt.title(f"Regression: {model_name}, peak_median_layer, area-{area}")
			plt.xlabel(f"bin width (ms)")
			plt.ylabel(f"$\\rho$")
			plt.ylim([0.0, 1.0])

		if save_tikz:
			filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-all-bin-widths-best-layer-local-tuned-{area}-{model_name}.tex")
			PlotterUtils.save_tikz(filepath)
		return data_dist


# ------------------  Violin plot: best layer of all networks ----------------#

	@staticmethod
	def voilin_plot_with_model_colors(data_dict, ax=None):
		"""Add voilin plot on the provided axis, using the data distribution 
		provided as a dictionary.

		Args:
			data_dict: dict = distribution of correlations with model name as key.
			ax: plt.axis = matplotlib axis to use for plotting.
		"""
		if ax is None:
			fig, ax = plt.subplots()
		# means = []
		x_coordinates = []
		# SEMs = []
		colors = []
		baseline_dist = data_dict['baseline']
		dist_list = []
		for i, (model_name, layer_data) in enumerate(data_dict.items()):
			dist_list.append(layer_data)
			# x_coordinates.append(model_name)
			# means.append(np.mean(layer_data))
			# SEMs.append(np.std(layer_data)/np.sqrt(len(layer_data))) #number of sents in test set?
			x_coordinates.append(i)
			colors.append(PlotterUtils.get_model_specific_color(model_name))

			
		vplot = ax.violinplot(
				dist_list, positions=x_coordinates,
				vert=True, showmedians=True,
				# showextrema=False
				# facecolor = colors
			)
		# adjusting colors and stuff...
		for i, pc in enumerate(vplot['bodies']):
			pc.set_facecolor(colors[i])
			pc.set_edgecolor('black')


		# ax.bar(x=x_coordinates, height=means, yerr=SEMs,
		#     color=colors)
		return ax

# ------------------  Bar plot: best layer of all networks ----------------#
	
	@staticmethod
	def bar_plot_with_model_colors(data_dict, ax=None):
		"""Add bar plot on the provided axis, using the data distribution 
		provided as a dictionary.

		Args:
			data_dict: dict = distribution of correlations with model name as key.
			ax: plt.axis = matplotlib axis to use for plotting.
		"""
		if ax is None:
			fig, ax = plt.subplots()
		means = []
		x_coordinates = []
		SEMs = []
		colors = []
		baseline_dist = data_dict['baseline']
		for model_name, layer_data in data_dict.items():
			x_coordinates.append(model_name)
			means.append(np.mean(layer_data))
			SEMs.append(np.std(layer_data)/np.sqrt(len(layer_data))) #number of sents in test set?
			colors.append(PlotterUtils.get_model_specific_color(model_name))

			
		ax.bar(x=x_coordinates, height=means, yerr=SEMs,
			color=colors)
		return ax
		
	@staticmethod
	def bar_plot_best_layer_all_networks(
			area='core', bin_width=20, threshold = 0.061,
			identifier = '_bins_corrected_100', p=0.01,
			marker_size=8, add_swarms=False, sig_marker_offset_y=-0.1
		):
		# area = args.area
		# bin_width = args.bin_width

		# model_names = PlotterUtils.model_names
		model_names = [
			'wav2letter_modified', 
			'deepspeech2', 'speech2text', 
			'whisper_tiny', 'wav2vec2',
			'whisper_base',
			
		]
		dist_peak_layer_each_model = {}
		colors = []
		colors.append(PlotterUtils.get_model_specific_color('baseline'))

		for model_name in model_names: 

			corr_obj = Correlations(model_name+identifier)
			# print(f"Object created for {model_name}, with id {identifier}")
			corr_dict_all_layers = corr_obj.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width,
				delay=0, threshold=threshold,
				normalized=True
			)
			layer_medians = {np.median(v):k for k,v in corr_dict_all_layers.items()}
			peak_median = max(layer_medians)
			peak_layer = layer_medians[peak_median]
			dist_peak_layer_each_model[model_name] = corr_dict_all_layers[peak_layer]
			colors.append(PlotterUtils.get_model_specific_color(model_name))
		
		# baseline is same for all networks..
		baseline_dist = corr_obj.get_baseline_corr_for_area(
			neural_area=area, threshold=threshold, bin_width=bin_width,
			delay=0, normalized=True
		)

		dist_all_models_and_baseline = {'baseline': baseline_dist, **dist_peak_layer_each_model}

		# plotting them...
		# ax = RegPlotter.bar_plot_with_model_colors(dist_peak_layer_each_model)
		ax = RegPlotter.voilin_plot_with_model_colors(dist_all_models_and_baseline)
		if add_swarms:
			swmplt = sns.swarmplot(
				dist_all_models_and_baseline,
				palette=colors,
				size=marker_size
				)
		
		
		plt.xticks(rotation=90, va='center', ha='center')
		# ax.axhline(np.median(corr_baseline), color='gray', ls='--')

		# statistical significance indicators
		# same baseline dist for every layer...
		dict_of_baseline_dist = {}
		for layer_ID in dist_peak_layer_each_model.keys():
			dict_of_baseline_dist[layer_ID] = baseline_dist

		RegPlotter.indicate_statistical_significance(
			dist_peak_layer_each_model,
			dict_of_baseline_dist,
			ax=ax, p = 0.01, size=15,
			offset_y=sig_marker_offset_y,
			offset_x=1,
			
		)

		for tick in plt.gca().get_xticklabels():
			tick.set_y(-0.25)

		plt.title(f"Reg, best layer-all networks, bw-{bin_width}ms, area-{area}")
		plt.xlabel(f"candidate models")
		plt.ylabel(f"$\\rho$")
		# plt.ylim([-0.0,1.0])

		filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-best-layer-all-networks-{area}-{bin_width}.tex")
		PlotterUtils.save_tikz(filepath)



	@staticmethod
	def indicate_statistical_significance(
			test_distributions,
			baseline_distributions=None,
			ax=None,
			p=0.01,
			size=15,
			offset_y = 0.8,
			offset_x = 0,
			color=None
		):
		"""Compares two iterables (dict) of distributions for statistical significance.
		Precisely, checks if each of the 'test_distributions' is statistically greater than
		the corresponding 'baseline_distributions'.
		
		Args:
			test_distributions: dict = distributions to be tested for significance,
				keys of this dict would be different models (networks/layers)
			baseline_distributions: dict = distributions taked as baseline, keys of 
				this dict MUST match that of 'test_distributions' for pairwise comparisons.
				if None (Default), test_distribution is treated as distribution of differences.
			p: float = significance threshold
			fontsize: int = value specifying size of * to be displayed.
			offset_y: float = vertical adjustment to the position of *
			offset_x: flaot = horizontal adjustment to the position of *
		"""
		if color is None:
			color = 'k'

		# ax.text specifies position bottom-left corner of text block, 
		# adjustment needed to center the stars at exact coordinates..
		# additionally, slight drift for networks with large number of layers,
		# was also observed, that is also being compensated.. 
		num_layers = len(test_distributions)
		font_adjustment = -0.01*((size/2.0) + (num_layers - 6)*1.3)

		sig_indices = []
		# baseline_dist = model_wise_distributions.pop('baseline')
		for i, (model_name, model_dist) in enumerate(test_distributions.items()):

			# if baseline_dist is None, test_dist is expected to be dist of differences..
			if baseline_distributions is None:
				baseline_dist=None
			else:
				# since test and baseline dicts must have the same keys..
				baseline_dist = baseline_distributions[model_name]

			stat_result = scipy.stats.wilcoxon(
				x = model_dist, 
				y = baseline_dist,
				alternative='greater', # tests only for x greater than y, (default was "two-sided")
			)
			print(f"p-value for {model_name}: {stat_result.pvalue}")
			if stat_result.pvalue < p:
				text_str = '*'
				sig_indices.append(i)


		heights = np.ones(len(sig_indices))*offset_y
		ax.scatter(sig_indices, heights, color=color, marker='*', s=size)

			# if stat_result.pvalue < p:
			#     text_str = '*'
			# else:
			#     text_str = ''

			
			# ax.text(
			#     i+offset_x+font_adjustment, offset_y, text_str,
			#     fontdict={'fontsize': size}, color=color
			# )

	@staticmethod
	def indicate_peak_and_similar_layers(layerwise_dist, p_threshold=0.01, offset_y=0.93):
		"""Indicates (in the existing figure) the peak layer (red star) 
		and all other layers statistcally not different from the peak leyer (black circles)
		
		Args:
			layerwise_dist: dict = dict of distributions across layers.
			p_threshold: float = p-value to be used for similarity, default=0.01.
			offset_y: int = vertical height of indicators (stars), default=0.93.

		"""
		medians = np.array([np.median(values) for values in layerwise_dist.values()])
		peak_median_layer = list(layerwise_dist.keys())[np.argmax(medians)]

		dist1 = layerwise_dist[peak_median_layer]

		statistically_same_layers = []
		colors = []
		for layer in layerwise_dist.keys():
			if layer != peak_median_layer:
				dist2 = layerwise_dist[layer]
				p_value = scipy.stats.wilcoxon(
								x = dist1, 
								y = dist2,
								alternative='two-sided', # tests if two distributions are significantly different...
							).pvalue

				if p_value > p_threshold:
					# means distributions are statistically the same.
					statistically_same_layers.append(layer)
					
		statistically_same_layers = np.array(statistically_same_layers)
		indicator_heights = np.ones_like(statistically_same_layers)*offset_y
		plt.scatter(statistically_same_layers, indicator_heights, c='k')

		plt.scatter([peak_median_layer], [offset_y], marker='*', c='r', s=100)
			

	@staticmethod
	def indicate_layer_architecture(ax, arch_specific_layer_ids, arch_ind_offset, arch_ind_lw):
		"""Indicates layer architecture by drawing a horizontal line 
		with architecture specific color, on top of layer-wise correlation 
		plots.
		"""
		for arch_type, layer_ids in arch_specific_layer_ids.items():
			heights = arch_ind_offset*np.ones_like(layer_ids)
			color = PlotterUtils.get_architecture_specific_color(arch_type)
			layer_ids = np.sort(layer_ids)
			# extend lines from -0.5 to +0.5 and remove space
			layer_ids[0] = layer_ids[0] - 0.5
			layer_ids[-1] = layer_ids[-1] + 0.5  

			ax.plot(layer_ids, heights, color=color, lw=arch_ind_lw)

# ------------------  saving regression results as pickle (for Makin) ----------------#

	@staticmethod
	def save_regression_correlations_for_model(
			model_name = 'deepspeech2',
			identifier='_bins_corrected_100',
			bin_width = 20
		):
		"""Saves regression correlations for the specified model,
		to pickle file. (in order to share with Makin) 
		"""
		corr_obj_trained = Correlations(model_name+identifier)
		threshold= corr_obj_trained.get_normalizer_threshold(
			bin_width=bin_width, poisson_normalizer=True
		)

		area = 'core'
		data_dist_core = corr_obj_trained.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width,
				delay=0, threshold=threshold,
				normalized=True
			)

		area = 'belt'
		data_dist_belt = corr_obj_trained.get_corr_all_layers_for_bin_width(
				neural_area=area, bin_width=bin_width,
				delay=0, threshold=threshold,
				normalized=True
			)

		model_results = {
			'core': data_dist_core,
			'belt': data_dist_belt
		}
		write_reg_corr(model_name, model_results)
	
