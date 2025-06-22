import os
import colorsys
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from palettable.colorbrewer import qualitative
from auditory_cortex import results_dir, valid_model_names

from utils_jgm.tikz_pgf_helpers import tpl_save


class PlotterUtils:
    """Contains utility methods for plotting.
    """
    model_names = valid_model_names[:6]
    colors = qualitative.Dark2_8.mpl_colors
    paired_colors = qualitative.Paired_12.mpl_colors

    color_adjustments = {
            'wav2vec2': (-0.2, 0.3),
            'whisper_tiny': (-0.2, 0.3),
            'whisper_base': (-0.2, 0.3),
        }
    
    architecture_types = ['transformer', 'conv', 'rnn']

    @classmethod
    def get_model_specific_color(cls, model_name):
        """Returns model specific color"""
        if model_name in cls.model_names:
            ind = cls.model_names.index(model_name)
            return cls.colors[ind]
        elif model_name == 'w2v2_audioset':
            return cls.paired_colors[5]
        elif model_name == 'cochresnet50':
            return cls.paired_colors[1]
        else:
            print(f"model_name '{model_name}' not recognizable!!!")
            return cls.colors[-1]
        
    @classmethod
    def get_architecture_specific_color(cls, layer_arch):
        """Returns color specific to the layer architecture e.g. cnn or rnn"""
        if layer_arch in cls.architecture_types:
            cmap = plt.colormaps['tab10']
            architecture_specific_colors = cmap(np.array([0, 5, 9]))
            ind = cls.architecture_types.index(layer_arch)
            return architecture_specific_colors[ind]

            # return cls.architecture_specific_colors[layer_arch]
        else:
            raise NameError(f"{layer_arch}: invalid layer architecure type.")
        
    @classmethod
    def get_model_specific_cmap(cls, model_name):
        """Retruns colormap specific to model_name, for heatmaps,
        it makes sure each colormap is around model specific color.
        
        """
        if model_name == 'speech2text':
            # this color is very close to the color of this model.
            model_color = "Oranges"
            cmap = sns.color_palette(model_color, as_cmap=True)
        elif model_name == 'deepspeech2':
            # this color is very close to the color of this model.
            model_color = "Greens"
            cmap = sns.color_palette(model_color, as_cmap=True)
        elif model_name == 'wav2letter_modified':
            # this color is very close to the color of this model.
            model_color = "Purples"
            cmap = sns.color_palette(model_color, as_cmap=True)
        elif model_name == 'wav2vec2':
            color = PlotterUtils.get_model_specific_color(model_name)
            # change color to make it more darker...
            hsv_color = mpl.colors.rgb_to_hsv(color)
            # adjust value...
            i = 2
            hsv_color[i] = hsv_color[i] - 0.2
            i = 1
            hsv_color[i] = hsv_color[i] + 0.3
            hsv_color = np.clip(hsv_color, 0, 1)
            color = mpl.colors.hsv_to_rgb(hsv_color)
            cmap = sns.light_palette(color, as_cmap=True)

        elif model_name == 'whisper_tiny':
            color = PlotterUtils.get_model_specific_color(model_name)
            # change color to make it more darker...
            hsv_color = mpl.colors.rgb_to_hsv(color)
            # adjust value...
            i = 2
            hsv_color[i] = hsv_color[i] - 0.2
            i = 1
            hsv_color[i] = hsv_color[i] + 0.3
            hsv_color = np.clip(hsv_color, 0, 1)
            color = mpl.colors.hsv_to_rgb(hsv_color)
            cmap = sns.light_palette(color, as_cmap=True)

        elif model_name == 'whisper_base':
            color = PlotterUtils.get_model_specific_color(model_name)
            # change color to make it more darker...
            hsv_color = mpl.colors.rgb_to_hsv(color)
            # adjust value...
            i = 2
            hsv_color[i] = hsv_color[i] - 0.25
            i = 1
            hsv_color[i] = hsv_color[i] + 0.3
            hsv_color = np.clip(hsv_color, 0, 1)
            color = mpl.colors.hsv_to_rgb(hsv_color)
            cmap = sns.light_palette(color, as_cmap=True)

        else:
            # for baseline...
            color = PlotterUtils.get_model_specific_color(model_name)
            cmap = sns.light_palette(color, as_cmap=True)

        return cmap
    
    @staticmethod
    def plot_spectrogram(
            spect, cmap=None
        ):
        """Plots spectrogram"""
        if cmap is None:
            cmap = 'viridis'
        plt.imshow(spect, origin='lower', interpolation=None,
           cmap=cmap)

        bin_width = 10
        xticks_step_ms = 400
        xticks_step_samples = int(xticks_step_ms/bin_width)
        total_bins = spect.shape[1]
        xticks = np.arange(0, total_bins, xticks_step_samples)

        # xticks = np.array([0, 40, 80, 120])
        yticks = np.array([0, 20, 40, 60 ])

        plt.xticks(xticks, bin_width*xticks)
        plt.yticks(yticks, yticks)
        plt.xlabel('time (ms)')
        plt.ylabel('mel filters')
        plt.title("Spectrogram for audio")

    @staticmethod
    def save_tikz(file_path):
        """Saves tikz plot to the file_path..."""
        # making sure the directory exists...
        dirpath = os.path.dirname(file_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        png_dir = os.path.join(dirpath, 'pngs')
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)

        extra_axis_parameters = {
            'width=\\figwidth',
            'height=\\figheight',
            'axis lines=left',
            'every x tick label/.append style={rotate=0}',
            'every axis plot/.append style={mark size=\\marksize}',
            'xticklabel style={opacity=\\thisXticklabelopacity, align=center}',
            'mark size=\\marksize',
        }
        tpl_save(
            filepath=file_path,
            extra_axis_parameters=extra_axis_parameters,
            tex_relative_path_to_data='pngs',
            extra_lines_start={
                '\\providecommand{\\figwidth}{5.7in}%',
                '\\providecommand{\\figheight}{2.0in}%',
                '\\providecommand{\\thisXticklabelopacity}{1.0}%',
                '\\providecommand{\\marksize}{2}%',
            },
        )
        print(f"result saved at: {file_path}")

    @staticmethod
    def create_cmap_using_rgb(rgb_colors_list):
        """"Takes in a list of rgb colors and returns a 
        colormap.

        Args:
            rgb_colors_list (list or adarray): (N,3) 
        """
        if not isinstance(rgb_colors_list, np.ndarray):
            rgb_colors_list = np.array(rgb_colors_list)
        alphas = np.ones((rgb_colors_list.shape[0],1))
        rgba = np.concatenate([rgb_colors_list, alphas], axis=1)
        colormap = mpl.colors.ListedColormap(rgba)
        return colormap
    
    @classmethod
    def create_cmap_using_hsl(cls, hsl_colors_list):
        """"Takes in a list of hsl colors and returns a 
        colormap.

        Args:
            hsl_colors_list (list or adarray): (N,3) 
        """
        if not isinstance(hsl_colors_list, np.ndarray):
            hsl_colors_list = np.array(hsl_colors_list)
        rgb_colors_list = np.zeros_like(hsl_colors_list)

        for i in range(hsl_colors_list.shape[0]):
            (h,l,s) = hsl_colors_list[i]
            rgb_colors_list[i] = colorsys.hls_to_rgb(h,l,s)
        colormap = cls.create_cmap_using_rgb(rgb_colors_list)
        return colormap


    @staticmethod
    def add_color_bar(
            ax, cmap, c_label='', **kwargs
        ):
        """Adds colorbar with parameters provided, creates
        space for colorbar on right side (8% space) of the
        provided axis, and returns the remaining space as an axis.
        
        Args:
            ax (plt.ax): Parent axis, inside which colorbar is
                to be added.
            cmap (str or plt.colormap): colormap for colorbar or
                str that can be converted to colormap.
            c_label (str): colorbar label.
        
        **kwargs:
            boundries (sequence): sequence of enteries 
            values (sequence): sequence of values, len(values)
                should be 1 less than len(boundries)

            Refer: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
            for understanding boundries & values. 

        Returns:
            pax (plt.axis): axis to the remaining space of the plot. 
        """
        if 'boundaries' in kwargs: boundaries = kwargs.pop('boundaries')
        else: 
            boundaries = None
        if 'values' in kwargs: values = kwargs.pop('values')
        else: 
            values = None
        if len(kwargs) != 0:
            raise ValueError('''You have provided unrecognizable keyword args''')
 
        # check if provided cmap is 
        if not isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            try:
                plt.get_cmap(cmap)
            except:
                raise ValueError("Provide a valid colormap.")

        plt.sca(ax)
        # axis to the remaining space.
        pax = plt.axes([0, 0, 0.9, 1.0])
        # axis for the colorbar 
        cax = plt.axes([0.92, 0.2, 0.04, 0.6])
        mappable = mpl.cm.ScalarMappable(cmap=cmap)
        plt.colorbar(
            mappable, cax=cax, label=c_label,
            boundaries=boundaries, values=values
        )
        return pax

    @staticmethod
    def split_transform(x, strt=0, stp = 1.0):
        """Implements piece-wise linear function such that
        values in (0, 0.5) are mapped to (0, 0.4) and those 
        in the range (0.5, 1.0) --> (0.6, 1.0), creating a
        split (hole) in the middle.  

        Args:
            x (float): value in the range (0.0, 1.0) to index 
                colormaps.
        """
        rangee = stp - strt
        hole = 0.1*rangee/(rangee) # 10% hold in the middle
        mid = (strt + stp)/2
        if x > mid:
            return (mid + hole) + (x - mid)*0.8, mid+(hole)*rangee, stp
        return (x - strt)*0.8 + strt, strt, mid-(hole)*rangee
    
    @classmethod
    def pre_cmap_transform(cls, x, num=2):
        """Transforms an input float (0, 1) using the piece-wise
        linear transform. The return value can be passed to any
        colormap, to get colormap with splits.
        """
        num -= 1
        out = cls.split_transform(x)
        while num>0:
            out = cls.split_transform(*out)
            num -= 1
        return out[0]
    
    @classmethod
    def add_color_bar_with_splits(cls, num_clusters, ax,
            cmap='viridis', c_label='', plain=False, split_levels=2
        ):
        """Adds colorbar with splits in between, for better
        qualitative viewing provided, creates space for colorbar on
        right side (8% space) of the provided axis, and returns the
        remaining space as an axis.
        
        Args:
            num_clusters (int): Number of qualitative level to
                display on colorbar.
            ax (plt.ax): Parent axis, inside which colorbar is
                to be added.
            cmap (str or plt.colormap): colormap for colorbar or
                str that can be converted to colormap.
            c_label (str): colorbar label.
            plain (bool): use splits on colormap or plain.
            split_levels (int): Number of discontinuous splits needed
                in the colormap, works well with 2 at max.
        """
        plt.sca(ax)
        # axis to the remaining space.
        pax = plt.axes([0, 0, 0.9, 1.0])
        # axis for the colorbar 
        cax = plt.axes([0.94, 0.15, 0.04, 0.7])

        values = np.arange(num_clusters+1)
        values = values/num_clusters
        if not plain:
            mapped_values = np.zeros_like(values)
            for i, val in enumerate(values):
                mapped_values[i] = cls.pre_cmap_transform(val, num=split_levels)
        else:
            mapped_values = values
        # plt.sca(ax)
        cax.imshow(np.atleast_2d(mapped_values).transpose(),
            extent=(0,2,0,num_clusters), cmap=cmap,
            vmin=0, vmax=1, origin='lower'
        )
        cax.set_xticks([])
        cax.set_title(c_label)
        return pax
        
    @staticmethod
    def get_dist_with_peak_median(dists_dict):
        """Given a dict of distributions, return the distribution
        having the max median.
        """
        dist_medians = {key: np.median(dist) for key, dist in dists_dict.items()}
        key_opt, peak_median = max(dist_medians.items(), key=lambda item: item[1])
        print(f"Peak median occurs at key={key_opt}")
        return dists_dict[key_opt]
    
    @staticmethod
    def plot_box_whisker_swarm_plot(
        distributions,
        color='green',
        width = 0.2,
        lw = 1.5,
        alpha = 0.2,
        ax = None,
        ):
        """Takes in a dictionary of distributions and plots box and
        whisker swram plot for each key: value pair in the dictionary"""
        dataframes = []
        for key, dist in distributions.items():
            dataframes.append(
                pd.DataFrame({
                    'Correlation': dist,
                    'areas': key})
            )
        data = pd.concat(dataframes)
        # Create the plot
        if ax is None:
            fig, ax = plt.subplots()
        # color = PlotterUtils.get_model_specific_color(model_name)
        if not isinstance(color, str):
            color = (*color, alpha)
        # Customize properties
        medianprops = {"color": "red", "linewidth": lw+0.5}
        boxprops = {"edgecolor": "black", "facecolor": color, "linewidth": lw}
        whiskerprops = {"color": "black", "linewidth": lw}
        capprops = {"color": "black", "linewidth": lw}

        ax = sns.boxplot(
            x='areas', 
            y='Correlation',
            data=data,
            width=width,
            medianprops=medianprops,
            boxprops=boxprops,
            capprops=capprops,
            whiskerprops=whiskerprops,
            ax=ax
            )
        sns.swarmplot(
            x='areas', y='Correlation', data=data, color=color, alpha=0.6,
            ax=ax)
        
        plt.ylabel(f"$\\Delta \\rho$")
        return ax

        # plt.show()