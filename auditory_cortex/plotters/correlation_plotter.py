import os
import numpy as np
import matplotlib.pyplot as plt

# local imports
from auditory_cortex import results_dir
from auditory_cortex.analyses import Correlations
from auditory_cortex.plotters.plotter_utils import PlotterUtils
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
        
        corr_obj = Correlations('wave2letter_modified_opt_neural_delay')
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
    def plot_line_with_shaded_region(data_dict, color, alpha=0.2,
            shaded_low_percentile=25, shaded_high_percentile=75,
            dotted_low_percentile=5, dotted_high_percentile=95,
            ax=None, display_dotted_lines=True
            
        ):

        if ax is None:
            fig, ax = plt.subplots()
        
        medians = []
        x_coordinates = []
        shaded_lower_percentiles = []
        shaded_higher_percentiles = []
        dotted_lower_percentiles = []
        dotted_higher_percentiles = []

        for layer_ID, layer_data in data_dict.items():
            medians.append(np.median(layer_data))
            x_coordinates.append(layer_ID)
            shaded_lower_percentiles.append(np.percentile(layer_data, shaded_low_percentile))
            shaded_higher_percentiles.append(np.percentile(layer_data, shaded_high_percentile))
            dotted_lower_percentiles.append(np.percentile(layer_data, dotted_low_percentile))
            dotted_higher_percentiles.append(np.percentile(layer_data, dotted_high_percentile))
        ax.plot(x_coordinates, medians, color=color)
        ax.fill_between(x=x_coordinates, y1=shaded_lower_percentiles, y2=shaded_higher_percentiles,
        alpha=alpha, color=color)
        # dotted lines...
        if display_dotted_lines:
            ax.plot(x_coordinates, dotted_lower_percentiles, '--', color=color)
            ax.plot(x_coordinates, dotted_higher_percentiles, '--', color=color)
        return ax

    @staticmethod
    def plot_all_network_layers_at_bin_width(model_name, area='core', bin_width=20,
                delay=0, threshold = 0.068, alpha=0.2, save_tikz=True, normalized=True,
                identifier='_sampling_rate_opt_neural_delay'):
        
        corr_obj = Correlations(model_name+identifier)
        data_dist = corr_obj.get_corr_all_layers_for_bin_width(
            neural_area=area, bin_width=bin_width,
            delay=delay, threshold=threshold,
            normalized=normalized
        )

        color = PlotterUtils.get_model_specific_color(model_name)
        ax=RegPlotter.plot_line_with_shaded_region(data_dict=data_dist,
                    color=color, alpha=alpha)
        plt.title(f"Regression: {model_name}, bw-{bin_width}ms, area-{area}")
        plt.xlabel(f"Layer IDs")
        plt.ylabel(f"$\\rho$")
        plt.ylim([0.0, 1.0])

        # plot baseline...
        area_sessions = corr_obj.metadata.get_all_sessions(area)
        baseline_dist = corr_obj.get_baseline_corr_session(
            sessions= area_sessions,bin_width=bin_width, delay=delay,
                    threshold=threshold, normalized=normalized)
        # same baseline dist for all layers...
        baseline_dist_all_layer = {}
        for layer_ID in data_dist.keys():
            baseline_dist_all_layer[layer_ID] = baseline_dist

        RegPlotter.plot_line_with_shaded_region(data_dict=baseline_dist_all_layer,
                    color='gray', alpha=alpha, ax = ax,
                    display_dotted_lines=False)

        if save_tikz:
            filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-layerwise-{area}-{model_name}.tex")
            PlotterUtils.save_tikz(filepath)

    @staticmethod
    def plot_one_network_layer_at_all_bin_width(model_name, area='core', layer=6,
                delay=0, threshold = 0.068, alpha=0.2, save_tikz=True,
                identifier='_sampling_rate_opt_neural_delay', labels=True,
                normalized=True):
        


        corr_obj = Correlations(model_name+identifier)
        data_dist = corr_obj.get_corr_all_bin_widths_for_layer(
            neural_area=area, layer=layer,
            delay=delay, threshold=threshold,
            normalized=normalized
        )

        RegPlotter.plot_line_with_shaded_region(data_dict=data_dist,
                    model_name=model_name, alpha=alpha)
        if labels:
            plt.title(f"Regression: {model_name}, layer-{layer}, area-{area}")
            plt.xlabel(f"Bin widths")
            plt.ylabel(f"$\\rho$")
            plt.ylim([0.0, 1.0])

        if save_tikz:
            filepath = os.path.join(results_dir, 'tikz_plots', f"Reg-all_bin_widths-layer{layer}-{area}-{model_name}.tex")
            PlotterUtils.save_tikz(filepath)


