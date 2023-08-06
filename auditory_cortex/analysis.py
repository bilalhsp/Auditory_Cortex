import os
import pickle
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
import matplotlib as mpl
import plotly.express as px
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from palettable.colorbrewer import qualitative


#local 
from auditory_cortex.models import Regression
from auditory_cortex.optimal_input import OptimalInput
import auditory_cortex.utils as utils
from auditory_cortex.utils import SyntheticInputUtils
from auditory_cortex import session_to_coordinates, CMAP_2D, session_to_subject, session_to_area
from auditory_cortex import saved_corr_dir, opt_inputs_dir
# from pycolormap_2d import ColorMap2DBremm, config, results_dir




# from naplib.visualization import imSTRF

class SyntheticInputs:
    def __init__(self, model_name, load_features=False):
        
        self.model_name = model_name
        self.wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')

        # default session for analysis
        self.session = '200206'
        self.betas = {}
        self.opt_objs = {}
        # self.colors = qualitative.Set2_8.mpl_colors
        self.colors = qualitative.Dark2_8.mpl_colors
        
        self.opt_objs[model_name] = OptimalInput(
            model_name, 
            load_features=load_features
        )
            
            # save betas using 'model+session' keys...
            # calling this will compute betas and store as attribute of opt_obj
            # _ = self.opt_objs[model_name].get_betas(self.session, use_cpu=False)

    
    def get_opt_obj(self, model_name=None):
        """Returns opt_obj for self.model_name, create one if it
        doesn't exist already."""
        if model_name is None:
            model_name = self.model_name
        if model_name not in self.opt_objs.keys():
            self.opt_objs[model_name] = OptimalInput(
                model_name, load_features=False
            )
        return self.opt_objs[model_name]
    
    def get_betas(self, session, model_name=None):
        """Compute or simply load betas (if already cached)"""
        session = str(int(session))
        if model_name is None:
            model_name = self.model_name
        beta_dir = os.path.join(opt_inputs_dir, model_name, 'betas', session)
        
        if not os.path.exists(beta_dir):
            print(f"Creating directory: {beta_dir}")
            os.makedirs(beta_dir)
        
        file_name = f"{model_name}_{session}_betas.npy"
        file_path = os.path.join(beta_dir, file_name)
        if os.path.exists(file_path):
            print("Loading betas..")
            betas = np.load(file_path)
        else:
            print("Computing betas...")
            self.opt_objs[model_name] = OptimalInput(
                model_name, load_features=True
            )
            betas = (self.get_opt_obj(model_name).get_betas(session)).cpu().numpy()
            np.save(file_path, betas)
        return betas

    def get_sessions_analyzed(self, model_name=None):
        """Returns list of sessions analyzed for the self.model_name"""
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        return os.listdir(wav_dir)
    
    def get_layers_analyzed(self, session, st_sent=None, model_name=None):
        """Returns the list of layers analyzed for the given session
        """
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name):
            layers = []
            channels = []
            sess_dir = os.path.join(wav_dir, session)
            filenames = os.listdir(sess_dir)
            
            for filename in filenames:
                # extracting starting sent
                sent_ind = filename.rfind('.wav')
                starting_sent = int(filename[sent_ind-3:sent_ind])
                
                if st_sent is not None and starting_sent != st_sent:
                    pass
                else:
                    # extracting layers and channels
                    ch_ind = filename.rfind('_corr')
                    ch = int(filename[ch_ind-2:ch_ind])
                    layer = int(filename[ch_ind-5:ch_ind-3])
                    layers.append(layer)

            return list(set(layers))

    def get_channels_analyzed(self, session, st_sent=None, layer=None, model_name=None):
        """Returns the list of channels analyzed for the given session
        """
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name):
            channels = []
            sess_dir = os.path.join(wav_dir, session)
            filenames = os.listdir(sess_dir)
            
            for filename in filenames:
                # extracting starting sent
                sent_ind = filename.rfind('.wav')
                starting_sent = int(filename[sent_ind-3:sent_ind])
                
                # extracting layers and channels
                ch_ind = filename.rfind('_corr')
                ch = int(filename[ch_ind-2:ch_ind])
                layer_analyzed = int(filename[ch_ind-5:ch_ind-3])
                if st_sent is not None and starting_sent != st_sent:
                    pass
                elif layer is not None and layer_analyzed != layer:
                    pass
                else:
                    channels.append(ch)       

            return list(set(channels))
        
    def get_st_sents_analyzed(self, session, model_name=None):
        """Returns the list of starting sents analyzed for the given session
        """
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name):
            sess_dir = os.path.join(wav_dir, session)
            filenames = os.listdir(sess_dir)
            starting_sents = []
            for filename in filenames:
                # extracting starting sent
                sent_ind = filename.rfind('.wav')
                starting_sent = int(filename[sent_ind-3:sent_ind])
                starting_sents.append(starting_sent)

            return list(set(starting_sents))
        
    def get_optimal_input(self, session, layer, ch, starting_sent, model_name=None):
        """Retrieves optimal waveform for given selection."""
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name): 
            sess_dir = os.path.join(wav_dir, session)
            file_found = False
            filenames = os.listdir(sess_dir)
            for filename in filenames:
                if f'{layer:02.0f}_{ch:02.0f}' in filename and f'starting_{starting_sent:03d}.wav' in filename:
                    select_file = filename
                    file_found = True
            if file_found:
                file_path = os.path.join(sess_dir, select_file)
                optimal_input = wavfile.read(file_path)
                corr_ind = select_file.rfind('_starting')
                corr = float(select_file[corr_ind-4 : corr_ind])
                return optimal_input[1], corr  
            else:
                raise FileNotFoundError(f"Optimal inputs not yet computed for layer-{layer}, ch-{ch}\
                                        with {starting_sent} as starting sent.")
        else:
            raise FileNotFoundError(f"Session-{session} not yet analyzed.")

    def plot_optimal_input(self, session, layer, ch, starting_sent, cmap='viridis', model_name=None,
                           ax=None):
        """plots spectrogram of optimal input for given selection."""
        if model_name is None:
            model_name = self.model_name
        synthetic_input, corr = self.get_optimal_input(session, layer, ch, starting_sent,
                                                       model_name=model_name)
        _, ax = SyntheticInputUtils.plot_spect(synthetic_input, ax=ax, cmap=cmap)

        ax.set_title(f" {model_name}, starting-{starting_sent} \n session-{session}, L-{layer}, ch-{ch}, $\\rho$={corr}")


    def analyze_synthetic_inputs(self, session, layer, st_sent=12,use_cpu=False, model_name=None):
        """Tabulates correlations for betas and optimal inputs for all 
        pair of channels and given layer."""

        session = str(int(session))
        # layer_idx = self.opt_objs[self.model_name].get_layer_index(layer)
        layer_idx = self.get_opt_obj(model_name).get_layer_index(layer)
        # layer_idx = self.opt_obj.linear_model.model_extractor.get_layer_index(layer)
        # betas = (self.get_opt_obj(model_name).get_betas(session)[layer_idx]).cpu().numpy()
        betas = self.get_betas(session, model_name)[layer_idx]

        df = pd.DataFrame(columns=['ch1', 'ch2', 'corr_opt_inputs','cross_corr_opt_inputs' ,'corr_betas'])
        channels = self.get_channels_analyzed(session, st_sent=st_sent, layer=layer,
                                              model_name=model_name)

        for ch1 in channels:

            opt1, _ = self.get_optimal_input(session, layer, ch1, starting_sent=st_sent,
                                             model_name=model_name)
            for ch2 in channels:
                if ch1 == ch2:
                    pass
                else:
                    opt2, _ = self.get_optimal_input(session, layer, ch2, starting_sent=st_sent, 
                                                     model_name=model_name)
                    corr_opt_inputs = utils.cc_single_channel(opt1, opt2)[0]
                    cross_corr_opt_inputs = np.max(np.correlate(opt1, opt2, mode='same'))
                    beta1 = betas[:, ch1]
                    beta2 = betas[:, ch2]
                    corr_betas = utils.cc_single_channel(beta1, beta2)[0]

                    df.loc[len(df.index)] = [ch1, ch2, corr_opt_inputs, cross_corr_opt_inputs, corr_betas]

        return df

    def plot_corr_vs_betas(self, session, layer, st_sent, ax=None, color=None,
                            model_name=None):

        if model_name is None:
            model_name = self.model_name
        if color is None:
            color = self.colors[0]
        df = self.analyze_synthetic_inputs(session, layer, st_sent, model_name=model_name)
        ax = df.plot.scatter(x='corr_betas', y='corr_opt_inputs', ax=ax, color=color)
        ax.set_title(f"{model_name}, \n session-{session}, L-{layer}, starting-{st_sent}")

    def plot_cross_corr_vs_betas(self, session, layer, st_sent, ax=None, color=None,
                                model_name=None):
        if model_name is None:
            model_name = self.model_name
        if color is None:
            color = self.colors[0]
        df = self.analyze_synthetic_inputs(session, layer, st_sent, model_name=None)
        ax = df.plot.scatter(x='corr_betas', y='cross_corr_opt_inputs', ax=ax, color=color)
        ax.set_title(f"{model_name}, \n session-{session}, L-{layer}, starting-{st_sent}")

    def analyze_synthetic_inputs_across_sessions(
            self, session1, session2, layer, st_sent=12, use_cpu=False, model_name=None
        ):
        """Tabulates correlations for betas and optimal inputs for all 
        pair of channels and given layer."""

        session1 = str(int(session1))
        session2 = str(int(session2))
        # layer_idx = self.opt_objs[self.model_name].get_layer_index(layer)
        layer_idx = self.get_opt_obj(model_name).get_layer_index(layer)
        # layer_idx = self.opt_obj.linear_model.model_extractor.get_layer_index(layer)
        # betas1 = (self.get_opt_obj(model_name).get_betas(session1)[layer_idx]).cpu().numpy()
        # betas2 = (self.get_opt_obj(model_name).get_betas(session2)[layer_idx]).cpu().numpy()
        betas1 = self.get_betas(session1, model_name)[layer_idx]
        betas2 = self.get_betas(session1, model_name)[layer_idx]
        df = pd.DataFrame(columns=['ch1', 'ch2', 'corr_opt_inputs','cross_corr_opt_inputs' ,'corr_betas'])
        channels1 = self.get_channels_analyzed(session1, st_sent=st_sent, layer=layer,
                                              model_name=model_name)
        channels2 = self.get_channels_analyzed(session2, st_sent=st_sent, layer=layer,
                                              model_name=model_name)

        for ch1 in channels1:

            opt1, _ = self.get_optimal_input(session1, layer, ch1, starting_sent=st_sent,
                                             model_name=model_name)
            for ch2 in channels2:
                opt2, _ = self.get_optimal_input(session2, layer, ch2, starting_sent=st_sent, 
                                                    model_name=model_name)
                corr_opt_inputs = utils.cc_single_channel(opt1, opt2)[0]
                cross_corr_opt_inputs = np.max(np.correlate(opt1, opt2, mode='same'))
                beta1 = betas1[:, ch1]
                beta2 = betas2[:, ch2]
                corr_betas = utils.cc_single_channel(beta1, beta2)[0]

                df.loc[len(df.index)] = [ch1, ch2, corr_opt_inputs, cross_corr_opt_inputs, corr_betas]

        return df


    def compare_sessions(self, session1, session2, layer, st_sent,
                          ax=None, model_name=None):
        if ax is None:
            fig, ax = plt.subplots()
        legend_elements = []
        # plotting session1
        self.plot_corr_vs_betas(session1, layer, st_sent, ax=ax, color=self.colors[0],
                                model_name=model_name)
        legend_elements.append(Line2D([0], [0], color=self.colors[0], label=f'{str(session1)}'))
        # plotting session1
        self.plot_corr_vs_betas(session2, layer, st_sent, ax=ax, color=self.colors[1],
                                model_name=model_name)
        legend_elements.append(Line2D([0], [0], color=self.colors[1], label=f'{str(session2)}'))
        
        # computing across sessions
        df = self.analyze_synthetic_inputs_across_sessions(
            session1, session2, layer, st_sent, model_name=None
        )
        legend_elements.append(Line2D([0], [0], color=self.colors[2], label=f'{str(session1)}-{str(session2)}'))
        # plotting across sessions..
        ax = df.plot.scatter(x='corr_betas', y='corr_opt_inputs', ax=ax, color=self.colors[2])
        ax.set_title(f"{model_name}, \n sessions-{session1} & {session2}, L-{layer}, starting-{st_sent}")
        ax.legend(handles=legend_elements, loc='best')



    def analyze_synthetic_inputs_across_models(
            self, model1, model2, session, layer1, layer2, st_sent):
        """Tabulates correlations for betas and optimal inputs for all 
        pair of channels and given layer."""

        session1 = str(int(session))

        # layer_idx = self.opt_objs[self.model_name].get_layer_index(layer)
        layer_idx1 = self.get_opt_obj(model1).get_layer_index(layer1)
        layer_idx2 = self.get_opt_obj(model2).get_layer_index(layer2)
        
        betas1 = self.get_betas(session, model1)[layer_idx1]
        betas2 = self.get_betas(session, model2)[layer_idx2]
        df = pd.DataFrame(columns=['ch1', 'ch2', 'corr_opt_inputs','cross_corr_opt_inputs' ,'corr_betas'])
        channels1 = self.get_channels_analyzed(session, st_sent=st_sent, layer=layer1,
                                              model_name=model1)
        channels2 = self.get_channels_analyzed(session, st_sent=st_sent, layer=layer2,
                                              model_name=model2)

        for ch1 in channels1:

            opt1, _ = self.get_optimal_input(session, layer1, ch1, starting_sent=st_sent,
                                             model_name=model1)
            for ch2 in channels2:
                opt2, _ = self.get_optimal_input(session, layer2, ch2, starting_sent=st_sent, 
                                                    model_name=model2)
                corr_opt_inputs = utils.cc_single_channel(opt1, opt2)[0]
                cross_corr_opt_inputs = np.max(np.correlate(opt1, opt2, mode='same'))
                beta1 = betas1[:, ch1]
                beta2 = betas2[:, ch2]
                corr_betas = utils.cc_single_channel(beta1, beta2)[0]

                df.loc[len(df.index)] = [ch1, ch2, corr_opt_inputs, cross_corr_opt_inputs, corr_betas]

        return df



    def compare_models(self, model_name1, model_name2, session, layer1, layer2, st_sent,
                          ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        legend_elements = []
        # plotting session1
        self.plot_corr_vs_betas(session, layer1, st_sent, ax=ax, color=self.colors[0],
                                model_name=model_name1)
        legend_elements.append(Line2D([0], [0], color=self.colors[0], label=f'{model_name1}'))
        # plotting session1
        self.plot_corr_vs_betas(session, layer2, st_sent, ax=ax, color=self.colors[1],
                                model_name=model_name2)
        legend_elements.append(Line2D([0], [0], color=self.colors[1], label=f'{model_name2}'))
        
        df = self.analyze_synthetic_inputs_across_models(
            model_name1, model_name2, session, layer1, layer2, st_sent)
        legend_elements.append(Line2D([0], [0], color=self.colors[2], label=f'{model_name1}-{model_name1}'))
        ax = df.plot.scatter(x='corr_betas', y='cross_corr_opt_inputs', ax=ax, color=self.colors[2])
        ax.set_title(f"{model_name1}, L-{layer1} & {model_name2}, L-{layer2} \n session-{session}, starting-{st_sent}")
        plt.legend(legend_elements)

        


class Correlations:
    def __init__(self, model_name=None, third=None) -> None:
        
        if model_name is None:
            model_name = 'wave2letter_modified'
        self.model = model_name
        filename = f'{model_name}_corr_results.csv'
        self.corr_file_path = os.path.join(saved_corr_dir, filename)
        
        # else:
        #     self.corr_file_path = corr_file_path
        self.data = pd.read_csv(self.corr_file_path)
        self.data['normalized_test_cc'] = self.data['test_cc_raw']/(self.data['normalizer'].apply(np.sqrt))
        # self.sig_threshold = sig_threshold

        # loading STRF baseline
        if third is None:
            STRF_filename = 'STRF_corr_results.csv'
        else:
            STRF_filename = f'STRF_{third}_third_corr_results.csv'

        STRF_file_path = os.path.join(saved_corr_dir, STRF_filename)
        self.baseline_corr = pd.read_csv(STRF_file_path)
        self.baseline_corr['strf_corr_normalized'] = self.baseline_corr['strf_corr']/(self.baseline_corr['normalizer'].apply(np.sqrt))
        # STRF_file_path = os.path.join(results_dir, 'cross_validated_correlations', 'STRF_corr_RidgeCV.npy')
        # self.baseline_corr = np.load(STRF_file_path)

        # using colorbrewer (palettable) colors... 
        colors = qualitative.Set2_8.mpl_colors
        # layer_types = self.data['layer_type'].unique()
        layer_types = ['conv', 'rnn', 'transformer']
        self.fill_color = {}
        for layer, color in zip(layer_types, colors):
            self.fill_color[layer] = color
        # self.fill_color = {'conv': Set2_3.mpl_colors[0], 'rnn': 'lightgreen', 'transformer': 'lightskyblue'}
        

    def plot_session_coordinates(self, threshold=None, dot_size=400, fontsize=12,
                                 subject_specific_color=False,
                                 core_belt_color=False, 
                                 ax = None):
        # fontsize = 12
        # dot_size = 400
        # cmap_2d = ColorMap2DZiegler(range_x=(-2, 2), range_y=(-2,2))
        sessions = self.get_significant_sessions(threshold=threshold)
        color_options = qualitative.Dark2_8.mpl_colors
        subject_color = {
                    'b': color_options[0],
                    'c': color_options[1],
                    'f': color_options[2]
                    }
        core_belt_colors = {
            'core': color_options[3],
            'belt': color_options[4],
            'parabelt': color_options[5]
                     
        }
        
        coordinates = []
        colors = []
        legend_handles = []
        for session in sessions:
            cxy = session_to_coordinates[int(session)]
            coordinates.append(cxy)
            if subject_specific_color:
                sub = session_to_subject[int(session)]
                colors.append(subject_color[sub])
            elif core_belt_color:
                area = session_to_area[int(session)]
                colors.append(core_belt_colors[area])
                
            else:  
            # colors.append(utils.coordinates_to_color(cmap_2d, cxy))
                colors.append(color_options[0])

        coordinates = np.array(coordinates)
        if ax is None:
            fig, ax = plt.subplots()
        circle = plt.Circle((0,0),2, fill=False)
        ax.set_aspect(1)
        ax.add_artist(circle)
        ax.set_xlim([-2.5,2.5])
        ax.set_ylim([-2.5,2.5])
        ax.set_title(f"coordinates color map", fontsize=fontsize)
        ax.set_xlabel('caudal - rostral (mm)', fontsize=fontsize)
        ax.set_ylabel('ventral - dorsal (mm)', fontsize=fontsize)
        
        plt.grid(visible=True)
        # size = size_scale*np.ones(len(colors))
        ax.scatter(coordinates[:,0], coordinates[:,1], c = colors, s=dot_size)
        return ax

    def get_baseline_corr_ch(self, session, ch, bin_width=20, delay=0, column='strf_corr'):
        corr = self.baseline_corr[
                (self.baseline_corr['session']==float(session))&\
                (self.baseline_corr['channel']==ch)&\
                (self.baseline_corr['bin_width']==bin_width)&\
                (self.baseline_corr['delay']==delay)
            ][column].head(1).item()
        return corr
    
    def get_baseline_corr_session(
            self, sessions=None, bin_width=20, delay=0, column='strf_corr', threshold=None):
                
        select_baseline = self.baseline_corr[
                (self.baseline_corr['bin_width']==bin_width)&\
                (self.baseline_corr['delay']==delay)
            ]
        
        if threshold is not None:
            select_baseline = select_baseline[
                (select_baseline['normalizer']>=threshold)
            ]
        if sessions is not None:
            # sessions should be list or None...
            session_baselines = []
            for session in sessions:
                session_baselines.append(select_baseline[
                        (select_baseline['session']==float(session))
                    ])
            select_baseline = pd.concat(session_baselines)    


        return select_baseline[column]

    def write_back(self):
        self.data.to_csv(self.corr_file_path, index=False)
    ##### TMPORARY function...to be removed 
    def add_layer_types(self):
        self.data['layer_type'] = 'conv'
        self.write_back

    def get_significant_sessions(self, threshold = None):
        """Returns sessions with corr scores above significant threshold for at least 1 channel"""
        if threshold is not None:
            # threshold = self.sig_threshold
            sig_data = self.data[self.data['normalizer'] >= threshold]
        else:
            sig_data = self.data
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
    
    def get_corr_score(self, session, layer, ch, bin_width=20, delay=0, N_sents=499):
        """Return the correlation coefficient for given specs."""
        select_data = self.data[
            (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']>=N_sents) &\
            (self.data['layer']==layer) &\
            (self.data['channel']==ch)   
            ]
        # print(select_data)
        return select_data.head(1)['test_cc_raw'].item()
    
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
    def get_good_channels(self, session, threshold=0.1,bin_width=20, delay=0, N_sents=499):
        """Return good channels for given session, layer and other selections.."""
        select_data = self.data[
            (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']>=N_sents) &\
            # (self.data['layer']==layer) &\
            (self.data['normalizer'] >= threshold)     
            ]
        return select_data['channel'].unique().tolist()

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
    
    def get_session_data(self, sessions=None, threshold=0.0,bin_width=20, delay=0, N_sents=499):
        """Returns session data for given settings"""
            
        select_data = self.data[
            # (self.data['session']==float(session)) & \
            (self.data['bin_width']==bin_width) & \
            (self.data['delay']==delay) & \
            (self.data['N_sents']>=N_sents) &\
            (self.data['normalizer'] >= threshold)     
            ]
        if sessions is not None:
            session_data = []
            # sessions is a list...
            for session in sessions:
                session_data.append(select_data[
                        (select_data['session']==float(session))
                    ])
            select_data = pd.concat(session_data)
            
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


    
    def box_plot_correlations(self, sessions=None, threshold=0.0,bin_width=20, delay=0, N_sents=499,
                    normalized=False, ax=None, delta_corr=False, y_axis_lim=None, lw=1.5):
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
            if y_axis_lim is not None:
                y_axis_lim += 0.3
        else:
            norm = ''
            column = 'test_cc_raw'
            strf_column = 'strf_corr'

        select_data = self.get_session_data(
            sessions, threshold=threshold, bin_width=bin_width, delay=delay, N_sents=N_sents
        )
        layer_ids = np.sort(select_data['layer'].unique())
        
        if delta_corr:
            layer_spread = {}
            for layer in layer_ids:
                differences = []
                ids = select_data[select_data['layer']==layer].index
                for id in ids:
                    channel_corr =  select_data.loc[id, column]
                    ch = int(select_data.loc[id, 'channel'])
                    sess = select_data.loc[id, 'session']
                    baseline = self.get_baseline_corr_ch(sess, ch, column=strf_column)
                    differences.append(channel_corr - baseline)

                layer_spread[int(layer)] = np.array(differences)
            y_axis_label = "$\Delta\\rho$"
        else:
            layer_spread = {}
            for layer in layer_ids:
                ids = select_data[select_data['layer']==layer].index
                layer_spread[int(layer)] = np.array(select_data.loc[ids, column]).squeeze()

            baseline_corr = self.get_baseline_corr_session(sessions, column=strf_column, threshold=threshold)
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
        median_lines = dict(color='k', linewidth=lw*2)  
        other_lines = dict(color='k', linewidth=lw)
        bplot = ax.boxplot(layer_spread.values(), positions = np.arange(1, len(layer_spread.keys())+1),
                    labels=layer_spread.keys(),
                    whis=[5,95],
                    capprops=other_lines,
                    whiskerprops=other_lines,
                    medianprops=median_lines,
                    patch_artist=True
                    )
        
        # setting the colors of the boxes as per layer type..
        for layer_id, box, flier in zip(layer_ids, bplot['boxes'], bplot['fliers']):
            ids = self.data[self.data['layer']==layer_id].index
            layer_type = self.data.loc[ids, 'layer_type'].unique().item()
            color = self.fill_color[layer_type]
            box.set(
                facecolor = color,
                linewidth=lw
            )
            flier.set(
                    markeredgecolor='k',
                    markerfacecolor=color,
            )

        ax.set_title(f"{self.model}, session-{sessions}{norm}")
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
        self.corr = Correlations()
    


