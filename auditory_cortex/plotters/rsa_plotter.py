import os
import numpy as np
import matplotlib.pyplot as plt


from auditory_cortex.analyses.deprecated.rsa import RSA
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex import results_dir
from utils_jgm.tikz_pgf_helpers import tpl_save


class RSAPlotter:

    @staticmethod
    def plot_line_with_shaded_region(data_dict, model_name, alpha=0.2, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        color = PlotterUtils.get_model_specific_color(model_name)
        means = []
        x_coordinates = []
        top_shaded = []
        bottom_shaded = []
        for layer_ID, layer_data in data_dict.items():
            layer_mean = np.mean(layer_data)
            layer_SEM = np.std(layer_data)#/np.sqrt(layer_data.size)
            
            x_coordinates.append(layer_ID)
            means.append(layer_mean)
            top_shaded.append(layer_mean + layer_SEM)
            bottom_shaded.append(layer_mean - layer_SEM)
        
        ax.plot(x_coordinates, means, color=color)
        ax.fill_between(x=x_coordinates, y1=bottom_shaded, y2=top_shaded,
        alpha=alpha, color=color)

        


    @staticmethod
    def RSA_plot_layer_wise(
            model_name, area='core', bin_width=20, 
            itr=100, identifier='global', alpha=0.2
        ):

        rsa = RSA(model_name=model_name, identifier=identifier)
        corr_dict = rsa.get_layer_wise_corr(
            area=area, bin_width=bin_width, iterations=itr, size=499
        )
        RSAPlotter.plot_line_with_shaded_region(corr_dict, model_name, alpha=alpha)
        plt.title(f"RSA, {model_name}, bw-{bin_width}ms, area-{area}")
        plt.xlabel(f"Layer IDs")
        plt.ylabel(f"$\\rho$")
        plt.ylim([-0.1,0.4])

        filepath = os.path.join(results_dir, 'tikz_plots', f"RSA-layerwise-{area}-{model_name}.tex")
        PlotterUtils.save_tikz(filepath)        

    @staticmethod
    def bar_plot_with_model_colors(data_dict, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        means = []
        x_coordinates = []
        SEMs = []
        colors = []
        for model_name, layer_data in data_dict.items():
            x_coordinates.append(model_name)
            means.append(np.mean(layer_data))
            SEMs.append(np.std(layer_data)/np.sqrt(layer_data.size))
            colors.append(PlotterUtils.get_model_specific_color(model_name))
            
        ax.bar(x=x_coordinates, height=means, yerr=SEMs,
            color=colors)
        
    # ------------------  Bar plot: best layer of all networks ----------------#
    @staticmethod
    def bar_plot_best_layer_all_networks(args):
        area = args.area
        bin_width = args.bin_width

        model_names = PlotterUtils.model_names

        dist_peak_layer_each_model = {}
        for model_name in model_names: 
            # for model_name in model_names:
            rsa = RSA(model_name=model_name)

            corr_dict = rsa.get_layer_wise_corr(
                area=area, bin_width=bin_width
            )
            layer_means = {np.mean(v):k for k,v in corr_dict.items()}
            peak_mean = max(layer_means)
            peak_layer = layer_means[peak_mean]

            dist_peak_layer_each_model[model_name] = corr_dict[peak_layer]

        # plotting them...
        RSAPlotter.bar_plot_with_model_colors(dist_peak_layer_each_model)
        plt.xticks(rotation=90, va='center', ha='center')

        for tick in plt.gca().get_xticklabels():
            tick.set_y(-0.25)

        plt.title(f"RSA, best layer-all networks, bw-{bin_width}ms, area-{area}")
        plt.xlabel(f"candidate models")
        plt.ylabel(f"$\\rho$")
        plt.ylim([-0.0,0.4])

        filepath = os.path.join(results_dir, 'tikz_plots', f"RSA-best-layer-all-networks-{area}-{bin_width}.tex")
        PlotterUtils.save_tikz(filepath)