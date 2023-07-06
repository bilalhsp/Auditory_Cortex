import os
import pickle
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
import matplotlib as mpl
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


#local 
# from auditory_cortex.analysis.config import *
import auditory_cortex.regression as Reg
# import auditory_cortex.analysis.config as config
import auditory_cortex.helpers as helpers
import auditory_cortex.utils as utils

from auditory_cortex import session_to_coordinates, CMAP_2D
from auditory_cortex import saved_corr_dir
# from pycolormap_2d import ColorMap2DBremm, config, results_dir


from scipy.signal import resample
from sklearn.linear_model import Ridge, ElasticNet
import naplib as nl
import math

from naplib.visualization import imSTRF

class STRF:
    def __init__(self, session):
        session = str(session)
        # self.reg_obj = helpers.get_regression_obj(load_features=False)
        self.reg_obj = Reg.transformer_regression(model_name = 'wave2letter_modified', load_features=False)
        # _ = self.reg_obj.load_spikes(bin_width=10, numpy=True)
        _ = self.reg_obj.get_neural_spikes(session, bin_width=20, numpy=True)
        # print(self.reg_obj.list_loaded_sessions())
        self.spikes = self.reg_obj.get_dataset_object(session).raw_spikes
        self.fs = self.reg_obj.dataset.fs

        sents = np.arange(1,499)
        self.random_sent_ids = np.random.permutation(sents)

    def get_sample(self, sent, ch=32):
        aud = self.reg_obj.dataset.audio(sent)
        # read spikes for the sent id, as (time, channel)
        spikes = self.spikes[sent].astype(np.float32)
        # get spectrogram for the audio inputs, as (time, freq) 
        spect = nl.features.auditory_spectrogram(aud, self.fs)
        # resample spect to get same # number of time samples as in spikes..
        spect = resample(spect, spikes.shape[0], axis=0)

        # spect has 128 channels at this point, we can reduce the channels 
        # for easy training, following examples lets reduce to 32 for now...
        spect_32 = resample(spect, 32, axis=1)

        return spect_32, spikes#np.expand_dims(spikes[:,ch], axis=1)
    
    def fit(self, estimator,num_workers=1):

        training_sent_ids = self.random_sent_ids[:350]
        
        # collecting data after pre-processing...
        train_spect_list = []
        train_spikes_list = []
        for sent in training_sent_ids:
            spect, spikes = self.get_sample(sent)
            train_spect_list.append(spect)
            train_spikes_list.append(spikes)
        
        # building a strf model...
        tmin = 0 # receptive field begins at time=0
        tmax = 0.3 # receptive field ends at a lag of 0.4 seconds
        sfreq = 100 # sampling frequency of data

        # estimator = Ridge(10, max_iter=max_itr)
        # setting show_progress=False would disable the progress bar
        strf_model = nl.encoding.TRF(tmin, tmax, sfreq, estimator=estimator,
                                    n_jobs=num_workers ,show_progress=True)

        # strf_model.fit(X=spects, y=spikess[:,32])
        # strf_model.fit(X=inp, y=out)
        strf_model.fit(X=train_spect_list, y=train_spikes_list)
        corr = self.evaluate(strf_model)


        return strf_model, corr
    
    def evaluate(self, model):
        testing_sent_ids = self.random_sent_ids[350:]
    
        # collecting data after pre-processing...
        test_spect_list = []
        test_spikes_list = []
        for sent in testing_sent_ids:
            spect, spikes = self.get_sample(sent)
            test_spect_list.append(spect)
            test_spikes_list.append(spikes)
        
        corr = model.corr(X=test_spect_list, y=test_spikes_list)
        return corr
    
        



class correlations:
    def __init__(self, model=None, sig_threshold=0.1) -> None:
        
        if model is None:
            model = 'wave2letter_modified'
        self.model = model
        filename = f'{model}_corr_results.csv'
        self.corr_file_path = os.path.join(saved_corr_dir, filename)
        
        # else:
        #     self.corr_file_path = corr_file_path
        self.data = pd.read_csv(self.corr_file_path)
        self.data['normalized_test_cc'] = self.data['test_cc_raw']/(self.data['normalizer'].apply(np.sqrt))
        self.sig_threshold = sig_threshold

        # loading STRF baseline
        STRF_file_path = os.path.join(saved_corr_dir, 'STRF_corr_results.csv')
        self.baseline_corr = pd.read_csv(STRF_file_path)
        self.baseline_corr['strf_corr_normalized'] = self.baseline_corr['strf_corr']/(self.baseline_corr['normalizer'].apply(np.sqrt))
        # STRF_file_path = os.path.join(results_dir, 'cross_validated_correlations', 'STRF_corr_RidgeCV.npy')
        # self.baseline_corr = np.load(STRF_file_path)
        
    def get_baseline_corr_ch(self, session, ch, bin_width=20, delay=0, column='strf_corr'):
        corr = self.baseline_corr[
                (self.baseline_corr['session']==float(session))&\
                (self.baseline_corr['channel']==ch)&\
                (self.baseline_corr['bin_width']==bin_width)&\
                (self.baseline_corr['delay']==delay)
            ][column].head(1).item()
        return corr
    
    def get_baseline_corr_session(
            self, session=None, bin_width=20, delay=0, column='strf_corr', threshold=None):
                
        select_baseline = self.baseline_corr[
                (self.baseline_corr['bin_width']==bin_width)&\
                (self.baseline_corr['delay']==delay)
            ]
        if session is not None:
            select_baseline = select_baseline[
                (select_baseline['session']==float(session))
            ]
        if threshold is not None:
            select_baseline = select_baseline[
                (select_baseline['normalizer']>=threshold)
            ]

        return select_baseline[column]

    def write_back(self):
        self.data.to_csv(self.corr_file_path)

    def get_significant_sessions(self, threshold = None):
        """Returns sessions with corr scores above significant threshold for at least 1 channel"""
        if threshold is None:
            threshold = self.sig_threshold
        sig_data = self.data[self.data['normalizer'] >= threshold]
        return sig_data['session'].unique()
    
    def get_all_sessions(self):
        """Returns all sessions in the saved results"""
        return self.data['session'].unique()
    
    def get_all_channels(self, session):
        """Return list of channels indices for given session."""
        return self.data[self.data['session'] == float(session)]['channel'].unique()
    def get_all_layers(self, session):
        """Return layers indices for given session."""
        return self.data[self.data['session'] == float(session)]['layer'].unique()
    
    def get_corr_score(self, session, layer, ch, bin_width=20, delay=0, N_sents=500):
        """Return the correlation coefficient for given specs."""
        select_data = self.data[
            (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']==N_sents) &\
            (self.data['layer']==layer) &\
            (self.data['channel']==ch)   
            ]
        return select_data['test_cc_raw'].item()
    
    def get_session_corr(self, session, bin_width = 20, delay = 0, N_sents = 499):
        """Returns correlations result for the specific 'session' and given selections
        """
        select_data = self.data[
            (self.data['session']==float(session)) &\
            (self.data['bin_width']==bin_width) &\
            (self.data['delay']==delay) &\
            (self.data['N_sents']>=N_sents)
            ]
        return select_data
    
    def session_bar_plot(
            self, 
            session = 200206,
            column = 'test_cc_raw', 
            cmap = 'magma', 
            ax = None, 
            separate_color_maps = True,
            vmin = 0,
            vmax = 1
            ):
        """Bar plots for session correlations (mean across channels for all layers)"""
        if ax is None:
            _, ax = plt.subplots()

        corr = self.get_session_corr(session)
        mean_layer_scores = corr.groupby('layer', as_index=False).mean()[column]
        num_layers = mean_layer_scores.shape[0]
        # print(mean_layer_scores.shape[0])
        if separate_color_maps:
            vmin = mean_layer_scores.min()
            vmax = mean_layer_scores.max()

        plt.imshow(np.atleast_2d(mean_layer_scores), extent=(0,num_layers,0,4), cmap=cmap, vmin=vmin, vmax=vmax)

    def topographic_bar_plots(
            self,
            figsize=10, 
            normalized=False, 
            threshold=0.1,
            separate_color_maps=True
        ):
        fig, axes = plt.subplots(figsize=(figsize,figsize))
        plt.grid(True)
        circle = plt.Circle((0,0),2, fill=False)
        axes.set_aspect(1)
        axes.add_artist(circle)
        axes.set_xlim([-2.0,2.0])
        axes.set_ylim([-2.0,2.0])
        # axes.set_title(f"Monkey, {monkey}")
        vmin = 0
        if normalized:
            column = 'normalized_test_cc'
            vmax = 1
        else:
            column = 'test_cc_raw'
            vmax = self.get_peak_corr(column=column)



        sessions = self.get_significant_sessions(threshold=threshold)
        sessions.sort()

        for session in sessions:
            cx, cy = session_to_coordinates[int(session)]
            cx = (cx + 2)/4 - 0.1
            cy = (cy + 2)/4 - 0.025
            ax = plt.axes([cx, cy, 0.2, 0.05])
            self.session_bar_plot(
                session, column=column, ax=ax, vmax=vmax, separate_color_maps=separate_color_maps
            )
            ax.set_title(session)
            ax.set_axis_off()


    def get_peak_corr(self, column, bin_width=20, delay=0, N_sents=499):
        
        select_data = self.data[
            (self.data['bin_width']==bin_width) &\
            (self.data['delay']==delay) &\
            (self.data['N_sents']>=N_sents)
            ]

        id = select_data.idxmax()[column]
        return self.data.iloc[id][column]
        

    def get_best_channel(self, session, layer, bin_width=20, delay=0, N_sents=500):
        """Returns channel id for max correlation with given data selection."""
        select_data = self.data[
            (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']==N_sents) &\
            (self.data['layer']==layer)   
            ]
        # id of highest correlation in the selection..!
        id = select_data.idxmax()['test_cc_raw']
        return self.data.iloc[id]['channel']
    def get_good_channels(self, session, layer, threshold=0.1,bin_width=20, delay=0, N_sents=500):
        """Return good channels for given session, layer and other selections.."""
        select_data = self.data[
            (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']==N_sents) &\
            (self.data['layer']==layer) &\
            (self.data['normalizer'] >= threshold)     
            ]
        return select_data['channel'].tolist()

    def summarize(self, session, threshold=0.0,bin_width=20, delay=0, N_sents=499,
                    col_name='test_cc_raw'):
        """Returns summary 'mean' and 'std' as function of layer for given session."""
        select_data = self.data[
            (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']>=N_sents) &\
            (self.data['normalizer'] >= threshold)     
            ]
        
        # std = select_data.groupby(['layer'])[col_name].describe()['std']
        # mean = select_data.groupby(['layer'])[col_name].describe()['mean']
        # max = select_data.groupby(['layer'])[col_name].describe()['max']
        # return mean, std, max
        return select_data.groupby(['layer'])[col_name].describe()
    
    def get_session_data(self, session=None, threshold=0.0,bin_width=20, delay=0, N_sents=499):
        """Returns session data for given settings"""
            
        select_data = self.data[
            # (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']>=N_sents) &\
            (self.data['normalizer'] >= threshold)     
            ]
        if session is not None:
            select_data = select_data[
                (select_data['session']==float(session))
            ]
            
        return select_data
    

    def get_selected_data(
                self, session=None, bin_width=None, delay=None, N_sents=None, threshold=None,
                layer=None, channel=None
            ):
        """Return selected data based on provided arguments. 
        If an argument if 'None', no filter is applied on that column."""
        select_data = self.data
        if session is not None:
            select_data = select_data[select_data['session']==float(session)]

        if bin_width is not None:
            select_data = select_data[select_data['bin_width']==float(bin_width)]
        
        if delay is not None:
            select_data = select_data[select_data['delay']==float(delay)]
        
        if N_sents is not None:
            select_data = select_data[select_data['N_sents']>=float(N_sents)]
        
        if threshold is not None:
            select_data = select_data[select_data['normalizer']>=float(threshold)]

        if layer is not None:
            select_data = select_data[select_data['layer']==float(layer)]


        if channel is not None:
            select_data = select_data[select_data['channel']==float(channel)]

        return select_data


    
    def box_plot_correlations(self, session=None, threshold=0.0,bin_width=20, delay=0, N_sents=499,
                    normalized=False, ax=None, delta_corr=False, y_axis_lim=None):
        """Plots box and whisker graphs for each layer of the given session, 
        if no session is mentioned, then it plots the same layer-wise plots by taking together 
        significant neurons from all sessions.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if normalized:
            norm = ', normalized'
            column = 'normalized_test_cc'
            strf_column = 'strf_corr_normalized'
            y_axis_lim += 0.3
        else:
            norm = ''
            column = 'test_cc_raw'
            strf_column = 'strf_corr'

        select_data = self.get_session_data(
            session, threshold=threshold, bin_width=bin_width, delay=delay, N_sents=N_sents
        )
        layer_ids = np.sort(select_data['layer'].unique())
        
        if delta_corr:
            layer_spread = {}
            # layer_ids = select_data['layer'].unique()
            # channel_ids = select_data['channel'].unique()
            for layer in layer_ids:
                differences = []
                ids = select_data[select_data['layer']==layer].index
                for id in ids:
                    channel_corr =  select_data.loc[id, column]
                    ch = int(select_data.loc[id, 'channel'])
                    sess = select_data.loc[id, 'session']
                    baseline = self.get_baseline_corr_ch(sess, ch, column=strf_column)
                    # if normalized:
                    #     baseline = baseline / select_data.loc[id, 'normalizer']
                    differences.append(channel_corr - baseline)

                layer_spread[int(layer)] = np.array(differences)
            y_axis_label = "$\Delta\\rho$"
        else:
            layer_spread = {}
            for layer in layer_ids:
                ids = select_data[select_data['layer']==layer].index
                layer_spread[int(layer)] = np.array(select_data.loc[ids, column]).squeeze()

            # layer_ids = select_data['layer'].unique()
            # for layer in layer_ids:
            #     layer_spread[int(layer)] =  np.array(select_data[select_data['layer']==layer][column])
            # # plotting the baseline...
            baseline_corr = self.get_baseline_corr_session(session, column=strf_column, threshold=threshold)
            print(f"Baseline median: {np.median(baseline_corr):.3f}")
            plt.axhline(baseline_corr.median(),
                        c='r', linewidth=3, ls='--', label='STRF baseline - (median)')
            plt.axhline(baseline_corr.max(),
                        c='r', linewidth=3, ls='--', alpha=0.3, label='STRF baseline - (peak)')
            plt.legend(loc='best')
            y_axis_label = "$\\rho$"
            if y_axis_lim is not None:
                ax.set_ylim([-0.05, y_axis_lim])

        # plotting function
        # ax1.set_title('Basic Plot')
        green_diamonds = dict(marker='D', markerfacecolor = 'g')
        red_lines = dict(color='r', linewidth=3)  
        ax.boxplot(layer_spread.values(), positions = np.arange(1, len(layer_spread.keys())+1), labels=layer_spread.keys(),
                    whis=[5,95], notch=True, flierprops = green_diamonds, medianprops=red_lines)

        ax.set_title(f"{self.model}, session-{session}{norm}")
        ax.set_xlabel('layer IDs')
        ax.set_ylabel(y_axis_label)
        # remove borders
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # # remove y-axis tick marsks
        # ax.yaxis.set_ticks_position('none')
        # add major grid lines in y-axis
        ax.grid(color='grey', axis='y', linestyle='-', linewidth = 0.5, alpha=0.5)

    
        return ax, layer_spread
    

    def plot_topographical_peaks(
            self,
            sessions = None,
            bin_width = 20,
            delay = 0,
            ax = None,
            threshold = 0.1,
            normalized = False,
            unit_circles = True,
            fontsize = 12,
            N_sents = 499,
            alpha=0.5
            
        ):
        """plots topographical peaks (based on coordinates of recording sites), 
        peak layer and corresponding correlation is worked out after taking median across 
        all significant channesl within each session, then for each recording site color of 
        the dot represents layer preference (peak median layer) and size (area) of the dot
        represents correlation strength.
        To help visualize the location of each recording site, a circle in plotted in the
        background that indicates the periphery of skull of the subject.
        Coordinates for c_LH have been reversed along x-axis (caudal-rostral) to map all 
        sessions (left and right hemispheres) onto the same coordinates."""


        if ax is None:
            fig, ax = plt.subplots()
        if normalized:
            norm = 'normalized'
            column = 'normalized_test_cc'
        else:
            norm = ''
            column = 'test_cc_raw'


        x_coordinates = []
        y_coordinates = []
        peak_layers = []
        peak_median_corr = []

        scale_size = 500
        sessions_info = ''
        if sessions is None:
            sessions = self.get_all_sessions()
            sessions_info = ', all sessions'
        # num_layers = len(self.get_all_layers('200206'))
        # num_layers = max(self.get_all_layers('200206'))
        num_layers = 11

        for session in sessions:
            select_data = self.get_session_data(
                session, threshold=threshold, bin_width=bin_width, delay=delay, N_sents=N_sents
            )
            if not select_data.empty:
                median_across_channels = select_data.groupby('layer', as_index=False).median()
                id = median_across_channels.idxmax()[column]

                # plots 'dot' with size (area) propotional to correlations,
                # and color as function of peak layer
                peak_median_corr.append(median_across_channels.loc[id, column]*scale_size)
                peak_layers.append(median_across_channels.loc[id, 'layer'])
                c_x, c_y = session_to_coordinates[int(session)]
                x_coordinates.append(c_x)
                y_coordinates.append(c_y)

        
        scatt = ax.scatter(
                    x_coordinates, y_coordinates, s=peak_median_corr, 
                    c=peak_layers, cmap='magma', vmin=0, vmax= num_layers, 
                )

        
        if unit_circles:         
            # adding circles of unit area..
            unit_areas = scale_size*np.ones(len(x_coordinates))
            ax.scatter(
                    x_coordinates, y_coordinates, s=unit_areas,
                    facecolor='none', edgecolor='black', alpha=alpha,
                )
        
        # formating plot and adding colorbar
        # adding background circle
        circle = plt.Circle((0,0),2, fill=False)
        ax.set_aspect(1)
        ax.add_artist(circle)
        ax.set_xlim([-2.5,2.5])
        ax.set_ylim([-2.5,2.5])
        ax.set_title(f"bin_width-{bin_width}ms, delay-{delay}{sessions_info}", fontsize=fontsize)
        ax.set_xlabel('caudal - rostral', fontsize=fontsize)
        ax.set_ylabel('ventral - dorsal', fontsize=fontsize)
        plt.grid(True)
        plt.colorbar(scatt, ax=ax, label='layers')

    def plot_coordinate_color_map(self, dot_size=400, fontsize=12):
        # fontsize = 12
        # dot_size = 400
        cmap_2d = ColorMap2DZiegler(range_x=(-2, 2), range_y=(-2,2))
        sessions = self.get_significant_sessions()
        coordinates = []
        colors = []
        for session in sessions:
            cxy = session_to_coordinates[int(session)]
            coordinates.append(cxy)
            colors.append(utils.coordinates_to_color(cmap_2d, cxy))

        coordinates = np.array(coordinates)
        fig, ax = plt.subplots()
        circle = plt.Circle((0,0),2, fill=False)
        ax.set_aspect(1)
        ax.add_artist(circle)
        ax.set_xlim([-2.5,2.5])
        ax.set_ylim([-2.5,2.5])
        ax.set_title(f"coordinates color map", fontsize=fontsize)
        ax.set_xlabel('caudal - rostral', fontsize=fontsize)
        ax.set_ylabel('ventral - dorsal', fontsize=fontsize)

        # size = size_scale*np.ones(len(colors))
        ax.scatter(coordinates[:,0], coordinates[:,1], c = colors, s=dot_size)



class PCA_analysis:
    def __init__(self, modes_file_path=None) -> None:
        if modes_file_path is None:
            modes_file_path = os.path.join(config.results_dir, config.pca_kde_sub_dir,
                                        config.pca_dist_modes_filename)
        self.data = pd.read_csv(modes_file_path)
        self.corr = correlations()
    


class PCA_topography:
    def __init__(self) -> None:
        # regression object and load features.
        # self.reg_obj = helpers.get_regression_obj('200206', load_features=False,
        #             checkpoint=checkpoint)
        self.reg_obj = Reg.transformer_regression(model_name='wave2letter_modified', load_features=False)
        self.features = None
        self.pcs = {}       # dict for principle components for layers...
        self.pca = {}       # dict for pca objects for layers (this can be used to get pcs for single sents)

        # correlation results object.
        self.corr = correlations()
        
        #saved pca_kde..
        self.kde_file_path = os.path.join(results_dir, pca_kde_sub_dir, pca_kde_data_filename)
        
        # if file already exists...
        if os.path.isfile(self.kde_file_path):
            print("Reading cached results...!")
            self.read_from_disk()    
        else:
            self.saved_kde_results = {}
            self.write_to_disk() 
        
    def read_from_disk(self):
        """Reads the current contents of the 'file'."""
        with open(self.kde_file_path, 'rb') as f:
            self.saved_kde_results = pickle.load(f)
    
    def write_to_disk(self):
        """Writes current status of 'saved_kde_results' to disk"""
        with open(self.kde_file_path, 'wb') as f:
            pickle.dump(self.saved_kde_results, f)   
    
    def get_significant_sessions(self):
        return self.corr.get_significant_sessions()

    def get_best_channel(self, session, layer):
        return self.corr.get_best_channel(session=session, layer = layer)
    
    def get_good_channels(self, session, layer, threshold=0.1):
        return self.corr.get_good_channels(session=session, layer = layer, threshold=threshold)
    
    def get_all_channels(self, session):
        return self.corr.get_all_channels(session)
    def get_corr_score(self, session, layer, ch):
        return self.corr.get_corr_score(session, layer, ch)
    

    def get_pcs(self, layer, bin_width=20, n_components=10):
        """Transforms features of the given layer onto 2d pc space
        (if not already done before) and returns the pcs.
        It also checks if features are pre-loaded, if not it loads them."""
        if self.features is None:
            self.reg_obj.load_features(bin_width=bin_width)
            self.features = self.reg_obj.unroll_features()
        pca = PCA(n_components=n_components)
        if layer not in self.pcs.keys():
            self.pcs[layer] = np.transpose(pca.fit_transform(self.features[layer]))
            self.pca[layer] = pca
        
        return self.pcs[layer] 

    def get_features(self, layer, bin_width=20):
        """Returns features for given layer, loads first if needed."""
        if self.features is None:
            # self.features = self.reg_obj.load_features(bin_width=bin_width, load_raw=True, numpy=True)
            self.reg_obj.load_features(bin_width=bin_width)
            self.features = self.reg_obj.unroll_features()
        return self.features[layer]

    def get_neural_spikes(self, session):
        """wraps the function of regression object."""
        spikes = self.reg_obj.get_neural_spikes(str(int(session)))
        return spikes

    def list_loaded_sessions(self):
        """Returns list of seession already loaded by regression object (self.obj)"""
        return self.reg_obj.list_loaded_sessions()

    def is_result_cached_already(self, session, layer, ch):
        session = str(int(session))
        if session in self.saved_kde_results.keys():
            return self.saved_kde_results[session]['layer-channel'][layer, ch] == 1
        return False    

    def get_near_peak_samples(self, session, layer, ch, sents=None, comps=None):
        """"Returns sample closest to the peak of kde plot, for each sent in sents.

        Args:
            session (int): ID of neural recording session.
            layer (int): ID of ANN layer.
            ch (int): ID of recording electrode/channel.
            sents (list): list of sent IDs that we want to analyze
            comps (list): 2 element list of PC's to be analyzed (plotted).
             
        Returns:
            near_peak_samples (dict): one near peak sample for each sent in sents. 

        """
        if comps is None:
            comps = [0,1]
        if sents is None:
            sents = np.arange(1,499)

        # getting the coordinates of peak in kde plot for given selection...
        z, *extent, peak_coordinates = self.compute_kde_2d(session, layer, ch, comps,
                                                        weighted=True, normalized=False)
        
        # for each sent, get the sample closest to the peak of kde plot...
        near_peak_samples = {}
        for sent in sents:
            # extract princple components for the given sentence...
            feats = self.reg_obj.raw_features[layer][sent]
            pc_sent = self.pca[layer].transform(feats)[:,comps]

            # get closest sample to the peak 
            num_samples = pc_sent.shape[0]
            distances = np.zeros(num_samples)
            for i in range(num_samples):
                distances[i] = math.dist(peak_coordinates, pc_sent[i,:])
            
            # add near peak sample to the dict
            near_peak_samples[sent] = np.argmin(distances)
        return near_peak_samples

    def get_audio_snippets_for_near_peak_samples(self, session, layer, ch, sents=None, comps=None, 
                                                time_bef=0.3, time_aft=0.1):


        near_peak_samples = self.get_near_peak_samples(session, layer, ch, sents, comps)
        # time_bef = 0.3  # 300 mili seconds
        # time_aft = 0.1  # 100 mili seconds
        snippet_widths = int((time_bef + time_aft)*16000)
        snippets = np.zeros((len(sents), snippet_widths))
        for i, sent in enumerate(sents):
            aud = self.reg_obj.dataset.audio(sent=sent)
            # from sample at 20ms rate, get sample at original fs=16000
            sample_for_peak = near_peak_samples[sent]*0.02*16000
            starting_sample = np.maximum(int(sample_for_peak - time_bef*16000), 0)
            ending_sample = np.minimum(int(sample_for_peak + time_aft*16000), aud.shape[0])

            snippets[i] = aud[starting_sample:ending_sample]

        return snippets

        

    
    def plot_kde_2d(self, session, layer, ch, comps=None, levels=None, color=None, ax=None,
                    normalized=True, weighted=True, threshold_factor=100):
        """
        plots contour plot for 2d kde for the given selection of session, layer and channel,
        
        Args:
            session (int): ID of neural recording session.
            layer (int): ID of ANN layer.
            ch (int): ID of recording electrode/channel.
            comps (list): 2 element list of PC's to be analyzed (plotted).
            levels (list): list of contours levels to be plotted.
            normalize (bool): if True plot is normalized by the unweighted KDE dist in
                    ANN space; Default: TRUE.
            weighted (bool): if True, plots KDE using neural spikes as weight, otherwise
                    plots KDE dist in ANN space without using neural spikes. This is ignored 
                    if normalize if True; Default: TRUE.
            color (str): color to be used for contour plot.
            ax (axes): axis to be used for contour plot.
        
        Returns:
            ax: matplotlib plotting axis.
            extent (list): Extent (range) of axis used for plotting. 

        """
        if color is None:
            colors = ['red']
        else:
            colors = [color]
        if levels is None:
            levels = [0.75, 0.80, 0.85, 0.9]
        if ax is None:
            _, ax = plt.subplots()
        # Do not save pcs, directly compute and plot...
        # z, extent = self.get_kde(session, layer, ch)

        z, *extent = self.compute_kde_2d(session, layer, ch, comps=comps,
                                            weighted=weighted, normalized=normalized,
                                            threshold_factor=threshold_factor)   
        
        x_min = extent[0]
        x_max = extent[1]
        y_min = extent[2]
        y_max = extent[3]
        m,n = z.shape
        X, Y = np.meshgrid(np.linspace(x_min,x_max,n), np.linspace(y_min, y_max,n), indexing='ij')
        # X, Y = np.mgrid[x_min:x_max:nj, y_min:y_max:nj]
        # # needed for plotting contours
        # if not normalized:
        #     z = z/z.max()         
        cs = ax.contour(X.T, Y.T, z/z.max(), levels=levels, colors=colors)
        return ax, extent
    
    def get_kde_2d(self, session, layer, ch, comps=None, weighted=True):
        """Returns saved kde results from the saved results, computes again, if not already saved."""
        session = str(int(session))
        layer = int(layer)
        ch = int(ch)
        if comps is None:
            comps = [0,1]

        pc_pairs = {
                [0,1]: 0,
                [0,2]: 1,
                [0,3]: 2,
                [1,2]: 3,
                [1,3]: 4,
                [2,3]: 5,
                    }
        if session not in self.saved_kde_results.keys():
            n_channels = self.get_all_channels(session).shape[0]
            self.saved_kde_results[session] = {
                            'kde_status': np.zeros((12, n_channels, len(pc_pairs))),
                            'null_kde_status': np.zeros((12, n_channels, len(pc_pairs))),
                            'kde': np.zeros((12,n_channels, len(pc_pairs), 100, 100)),
                            'null_kde': np.zeros((12,n_channels, len(pc_pairs), 100, 100)),
                            'extent': np.zeros((12,n_channels,len(pc_pairs), 4)),
                            'null_extent': np.zeros((12,n_channels,len(pc_pairs), 4)),
                        }
        pair_id = pc_pairs[comps]
        if weighted:
            if self.saved_kde_results[session]['kde_status'][layer, ch, pair_id] == 0:
                print("Compute kde (results not cached already)...!")
                z, *extent = self.compute_kde_2d(session, layer, ch, comps=comps, weighted=weighted, n=100)
                self.saved_kde_results[session]['kde'][layer,ch, pair_id,:,:] = z
                self.saved_kde_results[session]['extent'][layer, ch, pair_id,:] = extent
                self.saved_kde_results[session]['kde_status'][layer, ch, pair_id] = 1
                # write the updated resutls back to file..
                self.write_to_disk()
            else:
                z = self.saved_kde_results[session]['kde'][layer,ch,:,:]
                extent = self.saved_kde_results[session]['extent'][layer, ch,:]

        else:
            if self.saved_kde_results[session]['null_kde_status'][layer, ch, pair_id] == 0:
                print("Compute kde (results not cached already)...!")
                z, *extent = self.compute_kde_2d(session, layer, ch, comps=comps, weighted=weighted, n=100)
                self.saved_kde_results[session]['null_kde'][layer,ch, pair_id,:,:] = z
                self.saved_kde_results[session]['null_extent'][layer, ch, pair_id,:] = extent
                self.saved_kde_results[session]['null_kde_status'][layer, ch, pair_id] = 1
                # write the updated resutls back to file..
                self.write_to_disk()
            else:
                z = self.saved_kde_results[session]['null_kde'][layer,ch,:,:]
                extent = self.saved_kde_results[session]['null_extent'][layer, ch,:]

        return z, extent


    def compute_kde_2d(self, session, layer, ch, comps=None, weighted=True, normalized=True,
                    n=100, threshold_factor=100):
        """computes marginal distribution (kde) along the 2 principle components,
        mentionedc as comps.
        
        Args:
            session (int): ID of neural recording session.
            layer (int): ID of ANN layer.
            ch (int): ID of recording electrode/channel.
            comps (list): 2 element list of PC's to be analyzed (plotted).
            weighted (bool): if True, computed KDE using neural spikes as weight, otherwise
                    computes KDE dist in ANN space without using neural spikes; Default: TRUE.
            threshold: clipp null distribution (denominator) for numerical stability.

        Returns:
            z (ndarray): KDE distribution.
            extent (list): Extent (range) of PCs for dist. 
        """
        if comps is None:
            comps = [0,1]
        
        
        pcs = self.get_pcs(layer=layer, n_components=10)[comps]
        # n = pcs.shape[1]
        # d = len(comps)
        # kde_factor = n**(-1/(d+4))
        # print(kde_factor)
        # 100 points on both axis
        x_min = pcs[0].min()
        x_max = pcs[0].max()
        y_min = pcs[1].min()
        y_max = pcs[1].max()
        X, Y = np.meshgrid(np.linspace(x_min,x_max,n), np.linspace(y_min, y_max,n), indexing='ij')
        # X, Y = np.mgrid[x_min:x_max:nj, y_min:y_max:nj]
        positions = np.vstack([X.ravel(), Y.ravel()])

        if normalized:
            # getting the null distribution...
            kernel_null = scp.stats.gaussian_kde(dataset=pcs, weights=None)#, bw_method=kde_factor)
            values_null = kernel_null(positions)
            z_null = np.reshape(values_null, X.shape).T
            # in case of normalization, set weighted=True because we need both
            kde_factor = kernel_null.covariance_factor()


            spikes = self.reg_obj.get_neural_spikes(str(int(session)), numpy=True)
            weights=spikes[:,int(ch)]
            # getting spike weighted distribution...
            kernel = scp.stats.gaussian_kde(dataset=pcs, weights=weights, bw_method=kde_factor)
            values = kernel(positions)
            z = np.reshape(values, X.shape).T
            extent = x_min, x_max, y_min, y_max


            # threshold = 1.e-3
            # z_null_clipped = np.maximum(z_null, threshold)
            threshold = np.max(z_null)/threshold_factor
            z_null_clipped = z_null + threshold
            z_out = z/z_null_clipped
        else:
            if weighted:
                spikes = self.reg_obj.get_neural_spikes(str(int(session)), numpy=True)
                weights=spikes[:,int(ch)]
            else:
                weights=None

            # creating gaussian_kde object and getting values...
            kernel = scp.stats.gaussian_kde(dataset=pcs, weights=weights)#, bw_method=kde_factor)
            values = kernel(positions)
            z_out = np.reshape(values, X.shape).T

        # get the coordinates of peak kde plot, this will be used by some functions
        peak_coordinates = positions[:,np.argmax(values)]
        
        return z_out, x_min, x_max, y_min, y_max, peak_coordinates
    
    # def 
    
    def compute_kde_1d(self, session, layer, ch, weighted=True, comp=None):
        """compute kde for the principle components mentionedc as comps.."""
        if comp is None:
            comp = 0
        
        pc_1d = self.get_pcs(layer=layer, n_components=10)[comp]
        spikes = self.reg_obj.get_neural_spikes(str(int(session)))

        if weighted:
            weights=spikes[:,int(ch)]
        else:
            weights=None

        # 100 points on both axis
        x_min = pc_1d.min()
        x_max = pc_1d.max()
        positions = np.linspace(x_min, x_max, 100)

        
        # creating gaussian_kde object and getting values...
        kernel = scp.stats.gaussian_kde(dataset=pc_1d, weights=weights)
        kde = kernel(positions)

        return kde/np.sum(kde), x_min, x_max

    def plot_good_channels_for_session_and_layer(self, session, layer, levels=None, normalized=True, 
                                    corr_sign_threshold=0.1, ax=None, threshold_factor=100,
                                    legend_spacing=1, clrm='plasma', fontsize=22, comps=None, margin=0.2):
        if levels is None:
            levels = [0.7, 0.75, 0.8]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
        if comps is None:
            comps = [0,1]        
        cmap = plt.cm.get_cmap(clrm)
        # layer = 9
        # session = 200206
        # channels = pca_obj.get_all_channels(session)
        channels = self.get_good_channels(session, layer, corr_sign_threshold)

        N = len(channels)
        legend_elements = []
        for i, ch in enumerate(channels):    
            cs = self.plot_kde_2d(session, layer, ch, comps=comps, levels=levels,
                                  threshold_factor=threshold_factor,
                                  normalized=normalized, color=cmap(i/N), ax=ax)
            cc = self.get_corr_score(session, layer, ch)
            if i%legend_spacing==0:
                legend_elements.append(Line2D([0], [0], color=cmap(i/N), lw=4, 
                    label=f'ch-{ch}, \u0393-{cc:.2f}'))
        # # extracting axis limits..
        # for coll in ax.collections:
        #     xmax = -np.inf
        #     ymax = -np.inf
        #     xmin = np.inf
        #     ymin = np.inf
        #     for path in coll.get_paths():
        #         x_max, y_max = path.vertices.max(axis=0)
        #         xmax = max(x_max, xmax)
        #         ymax = max(y_max, ymax)
        #         x_min, y_min = path.vertices.min(axis=0)
        #         xmin = min(x_min, xmin)
        #         ymin = min(y_min, ymin)
        # xrange = xmax - xmin
        # yrange = ymax - ymin
        # margin /= 2         
        # ax.set_xlim(xmin - xrange*margin, xmax + xrange*margin)
        # ax.set_ylim(ymin - yrange*margin, ymax + yrange*margin)

        ax.set_xlabel(f"PC-{comps[0]}")
        ax.set_ylabel(f"PC-{comps[1]}")
            
        plt.title(f"All good channels for session-{session}, layer-{layer}", fontsize=fontsize)
        cax = plt.axes([0.95, 0.2, 0.04, 0.6])
        mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=sorted(channels))
        ax.legend(handles=legend_elements, loc='best')
        return ax
    
    def plot_significant_sessions_best_channel(self, layer, levels = None, fontsize=22,
                                            corr_sign_threshold=0.1, normalized=True,
                                            legend=False,
                                            ax=None,
                                            comps=None,
                                            margin=0.5,
                                            trim_axis=False,
                                            threshold_factor=100,
                                            exclude_session=None
                                            ):
        if comps is None:
            comps = [0,1]
        if levels is None:
            levels = [0.7, 0.75, 0.8]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
        # cmap = mpl.cm.get_cmap(clrm)
        sessions = self.get_significant_sessions()
        if exclude_session is not None:
            sessions = np.delete(sessions, np.where(sessions == exclude_session))


        N = len(sessions)
        print(f"N is: {N}")
        print(f"Session are:")
        print(sessions)
        legend_elements = []
        counter = 0
        for i, session in enumerate(sessions):
            channels = self.get_good_channels(session, layer, corr_sign_threshold)
            c_map = utils.get_2d_cmap(session)
            for k, ch in enumerate(channels):
            # ch = pca_obj.get_best_channel(session, layer)
                
                cc = self.get_corr_score(session, layer, int(ch))
                # cs = self.plot_kde(session, layer, ch, ax=ax, levels=levels, color=c_map, comps=comps)#cmap(counter/N))
                cs = self.plot_kde_2d(session, layer, ch, comps=comps, levels=levels,
                                  normalized=normalized, color=c_map, ax=ax, threshold_factor=threshold_factor)
                legend_elements.append(Line2D([0], [0], lw=4, color=c_map, 
                                    label=f"{int(session):6d},ch{int(ch):2d}, \u0393-{cc:.2f}"))
            counter += 1

        # # extracting axis limits..
        # if trim_axis:
        #     for coll in ax.collections:
        #         xmax = -np.inf
        #         ymax = -np.inf
        #         xmin = np.inf
        #         ymin = np.inf
        #         for path in coll.get_paths():
        #             x_max, y_max = path.vertices.max(axis=0)
        #             xmax = max(x_max, xmax)
        #             ymax = max(y_max, ymax)
        #             x_min, y_min = path.vertices.min(axis=0)
        #             xmin = min(x_min, xmin)
        #             ymin = min(y_min, ymin)
        #     xrange = xmax - xmin
        #     yrange = ymax - ymin
        #     margin /= 2         
        #     ax.set_xlim(xmin - xrange*margin, xmax + xrange*margin)
        #     ax.set_ylim(ymin - yrange*margin, ymax + yrange*margin)
        
        ax.set_xlabel(f"PC-{comps[0]}")
        ax.set_ylabel(f"PC-{comps[1]}")

        if legend:
            ax.legend(handles=legend_elements, loc='best')    
        ax.set_title(f"Contour plots for all significant sessions (best channels only): layer-{layer}")
        return ax



    def map_clusters_on_spect(self, layer, c1, c2, comps=None, sent=12,
                               ax=None, cmap='viridis'):
        """
        maps areas (clusters) of KDE in pc space, back onto the spectrogram 
        of given sent id.
        Args:   
            layer (int): layer id
            c1 (list): cluster 1 [xmin, xmax, ymin, ymax]
            c2 (list): cluster 2 [xmin, xmax, ymin, ymax]
            comps (list): princple component ids [pc1, pc2]\
            sent (int): sent id.
        """
        if comps is None:
            comps = [0,1]    
        # # Finding pc space 
        features = self.get_features(layer)
        pca = PCA(n_components=10)
        pc_10 = pca.fit_transform(features).transpose()

        # pc_ind = [0,2]
        feats_12 = self.reg_obj.extract_features(sents=[sent])
        pca_sent12 = pca.transform(feats_12[layer][sent]).transpose()

        s1 = np.where(pca_sent12[comps[0]] > c1[0])[0]
        s2 = np.where(pca_sent12[comps[0]] < c1[1])[0]
        s3 = np.where(pca_sent12[comps[1]] > c1[2])[0]
        s4 = np.where(pca_sent12[comps[1]] < c1[3])[0]

        s = np.intersect1d(s1, s2)
        s = np.intersect1d(s, s3)
        s = np.intersect1d(s, s4)

        s1 = np.where(pca_sent12[comps[0]] > c2[0])[0]
        s2 = np.where(pca_sent12[comps[0]] < c2[1])[0]
        s3 = np.where(pca_sent12[comps[1]] > c2[2])[0]
        s4 = np.where(pca_sent12[comps[1]] < c2[3])[0]

        r = np.intersect1d(s1, s2)
        r = np.intersect1d(r, s3)
        r = np.intersect1d(r, s4)

        audio = self.reg_obj.dataset.audio(sent=sent)
        spect = utils.spectrogram(audio)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))

        plt.imshow(spect, cmap=cmap, origin='lower')
        # fig = px.imshow(spect)
        handles = []

        color1 = 'red'
        color2 = 'black'
        for e in s:
            plt.axvline(x=e*2, color=color1)
            plt.axvline(x=(e+1)*2, color=color1)
            # fig.add_vrect(x0=e*2, x1=(e+1)*2, fillcolor=color1, opacity=0.2)
        handles.append(Line2D([0],[0], label='cluster-1', color=color1))
        for e in r:
            plt.axvline(x=e*2, color=color2)
            plt.axvline(x=(e+1)*2, color=color2)
            # fig.add_vrect(x0=e*2, x1=(e+1)*2, fillcolor=color2, opacity=0.2)
        handles.append(Line2D([0],[0], label='cluster-2', color=color2))
        plt.title(f"l-{layer}, pc{comps}, sent-{sent}")
        plt.legend(handles=handles)
        return ax

    def get_mode_of_marginal_dist(self, session, layer, ch, comp=None):
        """
        Computes mode of marginal distribtion along mentioend pc.
        """
        marginal, x_min, x_max = self.compute_kde_1d(session, layer, ch, comp=comp)
        x = np.linspace(x_min, stop=x_max, num=100)
        mode = (x[np.argmax(marginal)])
        mean = (np.mean(marginal*x))
        median = (np.median(marginal*x))

        return mode, mean, median
    
    def save_mode_of_marginal_dist(self, file_path, session, layer, ch, comp=None):
        if os.path.isfile(file_path):
            data = pd.read_csv(file_path)
        else:
            data = pd.DataFrame(columns=['session','layer','channel','pc', 'mean', 'median', 'mode'])

        mode, mean, median = self.get_mode_of_marginal_dist(session, layer, ch, comp=comp)

        data.loc[len(data.index)] = [session, layer, ch, comp, mean, median, mode]
        data.to_csv(file_path, index=False)


    def populate_df(self, session, layer):
        # if session not in self.dfs[layer]['session']:
        ch = self.corr.get_best_channel(session, layer)
        spikes = self.reg_obj.get_neural_spikes(str(int(session)))
        pcs = self.get_pcs(layer)
        # populate df 
        df1 = pd.DataFrame(columns = ['pc1', 'pc2', 'spike_rate', 'session', 'channel'],\
            dtype=np.float32)
        df1['pc1'] = pcs[:,0]
        df1['pc2'] = pcs[:,1]
        df1['spike_rate'] = spikes[:, int(ch)]
        df1['session'] = session*np.ones(df1.shape[0])
        df1['channel'] = ch*np.ones(df1.shape[0])
        
            # self.dfs[layer] = pd.concat([self.dfs[layer], df1])
        return df1


    
    def plot_kde_using_sns(self, session, ax=None, layer=0,c='r'):
        """Plots kde using dataframe and seaborn, computes kde for every plotting..."""
        df = self.populate_df(session, layer)
        print(f"Plotting session-{session}")
        sns.kdeplot(
                data=df,
                x='pc1',
                y='pc2',
                weights='spike_rate',
                levels=[0.75, 0.8, 0.85, 0.90],
                label=f'{session}',
                ax = ax
                )

    def plot_2d_colorbar(self, ax=None, n=100, add_circle=True):
        if ax is None:
            fig, ax = plt.subplots()
        x_min = -2
        x_max = 2
        y_min = -2
        y_max = 2
        cmap_2d = CMAP_2D(range_x=(x_min, x_max), range_y=(y_min, y_max))
        x_coordinates = np.linspace(x_min, x_max, n)
        y_coordinates = np.linspace(y_min, y_max, n)
        Y,X = np.meshgrid(x_coordinates, y_coordinates)
        coor = np.vstack([np.ravel(X), np.ravel(Y)])

        col = []
        for i in range(coor.shape[1]):
            # if (coor[0,i]**2 + coor[1,i]**2) > 4:
            #     col.append((1.0, 1.0, 1.0))
            # else:
            #     col.append(cmap(coor[0,i], coor[1,i])/255.0)
            # col.append(cmap(coor[0,i], coor[1,i])/255.0)
            col.append(self.coordinates_to_color(cmap_2d, coor[:,i]))

        plt.scatter(coor[0,:], coor[1,:], c=col)
        ax.set_title("2D colormap")

        if add_circle:
            circle =plt.Circle((0.0,0.0), 2, fill=False, linewidth = 3, edgecolor='b')
            ax.set_aspect(1)
            ax.add_artist(circle)
        ax.axis('off')
        # ax.set_xlim([-2.0,2.0])
        # ax.set_ylim([-2.0,2.0])

    def coordinates_to_color(self, cmap_2d, coordinates):
        return cmap_2d(coordinates[0], coordinates[1])/255.0

        
        
