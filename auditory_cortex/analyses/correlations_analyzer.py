import os
import yaml
import pickle
import numpy as np
import pandas as pd
import scipy as scp
# import seaborn as sns
import matplotlib as mpl
# import plotly.express as px
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from sklearn.decomposition import PCA

from palettable.colorbrewer import qualitative
from abc import ABC, abstractmethod

#local 
# from .normalizer_analyzer import NormalizerAnalyzer
from auditory_cortex.neural_data import NormalizerCalculator
from auditory_cortex.neural_data import create_neural_metadata

import auditory_cortex.utils as utils
# from auditory_cortex.utils import CorrelationUtils
# from auditory_cortex.neural_data import NeuralMetaData

# from auditory_cortex.neural_data.normalizer import Normalizer
# from auditory_cortex import session_to_coordinates, session_to_subject, session_to_area, area_to_sessions #, CMAP_2D
from auditory_cortex import saved_corr_dir, aux_dir, valid_model_names
from auditory_cortex.io_utils import io
# from pycolormap_2d import ColorMap2DBremm, config, results_dir

import logging
logger = logging.getLogger(__name__)


class BaseCorrelations(ABC):
    """Base class for Correlation and STRFCorrelation class. Provides
    the blueprint to create instances of sub-classes, merge results,
    add normalizer etc. """
    def __init__(self, model_name, dataset=None) -> None:
        
        self.model = model_name
        filename = f'{model_name}_corr_results.csv'
        self.corr_file_path = os.path.join(saved_corr_dir, filename)

        self.model_name = next(
            (model_name for model_name in valid_model_names if model_name in filename),
            None)

        if dataset is None:
            if 'ucdavis' in filename:
                self.dataset_name = 'ucdavis'
            else:
                self.dataset_name = 'ucsf'
        else:
            self.dataset_name = dataset

        
        self.metadata = create_neural_metadata(self.dataset_name)
        self.norm_obj = NormalizerCalculator(self.dataset_name)

        self.data = pd.read_csv(self.corr_file_path)

        ##############################################################
        ###     The following might not be needed anymore! (06-26-25)
        ##############################################################
        # check if 'test_cc_raw' not one of the columns
        # this will be the case for STRF correlations
        # in that case, copy 'strf_corr' to 'test_cc_raw'
        if 'test_cc_raw' not in self.data.columns:
            self.data['test_cc_raw'] = self.data['strf_corr']
            logger.info(f"'test_cc_raw added as a column..!")
        
        if 'normalizer' not in self.data.columns:
            self.data.loc[:, 'normalizer'] = np.nan
            # self.set_normalizers()
            self.set_normalizers_using_bootsrap()

        if 'normalizer_app' not in self.data.columns:
            # need to remove these bad channels...
            bad_sessions = [180725., 190607., 191212., 200226.]
            all_bad_ids = []
            for sess in bad_sessions:
                all_bad_ids.append(self.data[self.data['session'] == sess].index)
            all_bad_ids = np.concatenate(all_bad_ids)
            self.data.drop(all_bad_ids, inplace=True)


    @abstractmethod
    def get_selected_data(self, *args, **kwargs):
        """Abstract method, any derived class must implement this."""
        pass

    @staticmethod
    def get_highly_tuned_channels(select_data, threshold, mVocs):
        norm_column = 'normalizer'
        null_mean_col = 'null_mean'
        null_std_col = 'null_std'
        if mVocs:
            norm_column = 'mVocs_'+norm_column
            null_mean_col = 'mVocs_'+null_mean_col
            null_std_col = 'mVocs_'+null_std_col

        # if mVocs:
        #     norm_column = 'mVocs_'+norm_column
        # print(f"Applying threshold: {threshold:.3f} on column: '{norm_column}'...")
        # select_data = select_data[select_data[norm_column]>=float(threshold)]
        logger.info(f"Filtering '{norm_column}' using multiple of {threshold:.3f} with std dev ...")
        select_data = select_data[
            (select_data[norm_column] > 0.0) & \
            (select_data[norm_column] > select_data[null_mean_col] + threshold*select_data[null_std_col])
            ]
            
        return select_data


    # ---------------- methods using normalizer dist. using all-possible-pairs (app) ---------#
    def set_normalizers_using_bootsrap(self, mVocs=False, norm_bin_width=None, verbose=False):
        """Reads normalizer distributions (both True & Null) from the memory and
        incorporates them to the current file by following steps:
            - add the following columns (filled by appropriate values):
                + 'normalizer': mean of the normalizer distribution.
                + 'null_mean': mean of the null distribution.
                + 'null_std': std of the null distribution.
            - adds a columns of normalized correlations:
                + 'normalized_test_cc': test_cc_raw / sqrt(normalizer)
            In case of mVocs, all the columns are prefixed with 'mVocs_'.

        Args:
            mVocs: bool = IF True, update normalizer for monkey vocalizations, 
                else, update normalizer for Timit stimuli.
            norm_bin_width: int = bin width in ms. Default=None. When specified,
                this copies normalizer at norm_bin_width to the results at all bin widths. 
                This will be used when we are predicting at fixed bin width, but features
                were low-pass-filtered at different bin widths.
        """
        raw_corr_col = 'test_cc_raw'
        norm_corr_col = 'normalized_test_cc'
        normalizer_col = 'normalizer'
        null_mean_col = 'null_mean'
        null_std_col = 'null_std'
        if mVocs:
            raw_corr_col = 'mVocs_'+raw_corr_col
            norm_corr_col = 'mVocs_'+norm_corr_col
            normalizer_col = 'mVocs_'+normalizer_col
            null_mean_col = 'mVocs_'+null_mean_col
            null_std_col = 'mVocs_'+null_std_col

        bin_widths = self.data['bin_width'].unique()
        for bin_width in bin_widths:
            select_data = self.get_selected_data(bin_width=bin_width)
            sessions = select_data['session'].unique()
            # delays = select_data['delay'].unique()
            for session in sessions:
                # for delay in delays:
                select_data = self.get_selected_data(
                    sessions=[session], bin_width=bin_width, #delay=delay
                )
                channels = select_data['channel'].unique()

                # reading the normalizer...
                if norm_bin_width is None:
                    bw_norm = bin_width
                else:
                    bw_norm = norm_bin_width

                norm_dist, null_dist = self.norm_obj.get_inter_trial_corr_dists_for_session(
                    session=session, bin_width=bw_norm, mVocs=mVocs,
                    )

                # norm_means = np.mean(norm_dist, axis=0)
                # null_means = np.mean(null_dist, axis=0)
                # null_std = np.std(null_dist, axis=0)

                for ch in channels:
                    # ch = int(ch)
                    # ch_normalizer = norm_means[ch]
                    # ch_null_mean = null_means[ch]
                    # ch_null_std = null_std[ch]

                    ch_normalizer = np.mean(norm_dist[ch])
                    ch_null_mean = np.mean(null_dist[ch])
                    ch_null_std = np.std(null_dist[ch])

                    ids = select_data[select_data['channel']==ch].index
                    self.data.loc[ids, normalizer_col] = ch_normalizer
                    self.data.loc[ids, null_mean_col] = ch_null_mean
                    self.data.loc[ids, null_std_col] = ch_null_std 

        if 'layer_type' not in self.data.columns and self.model_name is not None:
            config = utils.load_dnn_config(self.model_name)
            num_layers = len(config['layers'])
            layer_types = {
                config['layers'][i]['layer_id']: config['layers'][i]['layer_type'] 
                for i in range(num_layers)
                }
            for layer_id in layer_types.keys():
                layer_type = layer_types[layer_id]
                ids = self.data[self.data['layer']==layer_id].index
                self.data.loc[ids, 'layer_type'] = layer_type

        self.data[norm_corr_col] = self.data[raw_corr_col]/(self.data[normalizer_col].apply(np.sqrt))
        logger.info(f"Columns: '{normalizer_col}', '{raw_corr_col}', '{norm_corr_col}' updated using normalizer (random pairs) dist, writing back now...")
        self.write_back()
        #                 self.data.loc[ids, 'normalizer'] = ch_normalizer 



# ------------------  Retrieve data for analysis ----------------#
    def get_normalizer_threshold(
            self, bin_width=20, poisson_normalizer=True, mVocs=False,
            threshold_percentile=None
        ):
        """Retrieves significance threshold, 90th percentile of the NULL distribution
        of normalizer.

        Args:
            bin_width: int = bin width in ms
            poisson_normalizer: bool = If True, use NULL distriubtion 
                based on random Poisson sequences.
            mVocs: bool = If True, estimate threshold for monkey voc. 
                test set, else for timit test set. The difference 
                in the duration of test set stimuli is used to determine 
                the length of random sequences.
        """
        # threshold_dict = io.read_normalizer_threshold(
        #     bin_width=bin_width, poisson_normalizer=poisson_normalizer
        #     )
        # if threshold_dict is None or bin_width not in threshold_dict.keys():
        if threshold_percentile is None:
            threshold_percentile=90
        if poisson_normalizer:
            null_dist = self.norm_obj.get_normalizer_null_dist_using_poisson(
                bin_width=bin_width, mVocs=mVocs,
                )
            threshold = np.percentile(null_dist, threshold_percentile)    
            # threshold = self.norm_obj.compute_normalizer_threshold_using_poisson(bin_width=bin_width)[0]
        else:
            threshold = self.norm_obj.compute_normalizer_threshold(bin_width=bin_width)[0]
        return threshold



    # # ------------------  copy normalizer for all bin-widths ----------------#
    # def set_normalizers(self, bin_widths: list=None):
    #     """Set normalizers (repeatability correlations) from the saved results.
    #     """
    #     if bin_widths is None:
    #         bin_widths = self.data['bin_width'].unique()
    #     for bin_width in bin_widths:
    #         select_data = self.get_selected_data(bin_width=bin_width)
    #         sessions = select_data['session'].unique()
    #         delays = select_data['delay'].unique()
    #         for session in sessions:
    #             for delay in delays:
    #                 select_data = self.get_selected_data(
    #                     sessions=[session], bin_width=bin_width, delay=delay
    #                 )
    #                 channels = select_data['channel'].unique()
    #                 selected_normalizers = self.norm_obj.get_normalizer_for_session(
    #                     session=session, bin_width=bin_width, delay=delay
    #                     )
    #                 for ch in channels:
    #                     ids = select_data[select_data['channel']==ch].index
    #                     ch_normalizer = selected_normalizers[
    #                             selected_normalizers['channel']==ch
    #                         ]['normalizer'].head(1).item()
    #                     self.data.loc[ids, 'normalizer'] = ch_normalizer #/0.82   # adjusting for context..

    #     self.data['normalized_test_cc'] = self.data['test_cc_raw']/(self.data['normalizer'].apply(np.sqrt))
    #     print(f"Normalizers updated FOR CONTEXT AS WELL, writing back now...")
    #     self.write_back()

    

    def write_back(self):
        """Saves the updated dataframe to disk.
        """
        self.data.to_csv(self.corr_file_path, index=False)
        logger.info(f"Saved at {self.corr_file_path}")

    def get_filepath(self):
        """Returns abs path of corr result file."""
        return self.corr_file_path


class STRFCorrelations(BaseCorrelations):
    def __init__(
            self, model_name = None
            
        ) -> None:
        if model_name is None:
            model_name = f'STRF_freqs80_all_lags'

        super().__init__(model_name)



    def get_significant_data_using_statistical_inclusion(
                self, bin_width, sessions: list=None, delay=None, lag=None,
                inclusion_p_threshold=0.01, use_poisson_null=True, mVocs=False
        ):
        """Retrieves selected data based on provided arguments. 
        If an argument if 'None', no filter is applied on that column.
        Only retrieves data for signigicant neurons (selected based on 
        t-test by normalizer object.)

        Args:
            bin_width: int = bin_width in ms.
            sessions: list of ints = list of sessions IDs to get data for,
                if no session is provided, all significant sessions are retrieved.
            delay: int = delay in ms.
            lag: int = max delay spanned by the TRF window.

        
        Returns:
            pandas DataFrame
        """
        select_data = self.data
        select_data = select_data[select_data['bin_width']==float(bin_width)]
        
        if delay is not None:
            select_data = select_data[select_data['delay']==float(delay)]

        if lag is not None:
            select_data = select_data[select_data['tmax']==float(lag)]

        if use_poisson_null:
            sig_sess_n_chs = self.norm_obj.get_significant_sessions_and_channels_using_poisson_null(
                bin_width=bin_width, p_threshold = inclusion_p_threshold, mVocs=mVocs
                )
        else:
            sig_sess_n_chs = self.norm_obj.get_significant_sessions_and_channels_using_shifts_null(
                bin_width=bin_width, p_threshold = inclusion_p_threshold, mVocs=mVocs
                )

        if sessions is None:
            sessions = list(sig_sess_n_chs.keys())

        filtered_data = []
        # sessions is a list...
        for session in sessions:
            session = str(int(float(session)))
            if session in sig_sess_n_chs.keys():
                session_data = select_data[(select_data['session']==float(session))]
                for ch in sig_sess_n_chs[session]:
                    filtered_data.append(
                        session_data[session_data['channel']==float(ch)]
                    )

        filtered_data = pd.concat(filtered_data)
        return filtered_data
    
    def get_selected_data(
                self, sessions: list=None, bin_width=None, delay=None, 
                threshold=None, lag=None, mVocs=False
        ):
        """Retrieves selected data based on provided arguments. 
        If an argument if 'None', no filter is applied on that column.

        Args:
            sessions: list of ints = list of sessions IDs to get data for.
            bin_width: int = bin_width in ms.
            delay: int = delay in ms.

            mVocs: bool = If True, use 'mVocs_normalizer' for significance
        
        Returns:
            pandas DataFrame
        """
        select_data = self.data
        if bin_width is not None:
            select_data = select_data[select_data['bin_width']==float(bin_width)]
        
        if delay is not None:
            select_data = select_data[select_data['delay']==float(delay)]
        
        # if channel is not None:
        #     select_data = select_data[select_data['channel']==float(channel)]
        if threshold is not None:
            select_data = self.get_highly_tuned_channels(select_data, threshold=threshold, mVocs=mVocs)
            # norm_column = 'normalizer'
            # if mVocs:
            #     norm_column = 'mVocs_'+norm_column
            # print(f"Applying threshold: {threshold:.3f} on column: '{norm_column}'...")
            # select_data = select_data[select_data[norm_column]>=float(threshold)]

        if lag is not None:
            select_data = select_data[select_data['tmax']==float(lag)]

        if sessions is not None:
            session_data = []
            # sessions is a list...
            for session in sessions:
                session_data.append(select_data[
                        (select_data['session']==float(session))
                    ])
            select_data = pd.concat(session_data)
        return select_data


    def get_correlations_for_bin_width(
            self, neural_area='core', bin_width=20, delay=0,
            threshold=None, lag=None, normalized=True, mVocs=False,
            use_stat_inclusion=False, inclusion_p_threshold=0.01,
            use_poisson_null=True,
        ):
        """Retrieves the column of correlations (normalized or un-normalized), for the
        given selections filters.
        """
        
        column='test_cc_raw'
        if normalized:
            # Deprecated...
            # if use_stat_inclusion:
            #     column = 'corr_normalized_app'
            # else:
            column = 'normalized_test_cc'

        if mVocs:
            column = 'mVocs_'+column
        # if lag is None:
        #     lag = 0.3

        area_sessions = self.metadata.get_all_sessions(neural_area)
        if use_stat_inclusion:
            selected_data = self.get_significant_data_using_statistical_inclusion(
                sessions=area_sessions, bin_width=bin_width, delay=delay,
                lag=lag, inclusion_p_threshold=inclusion_p_threshold,
                use_poisson_null=use_poisson_null, mVocs=mVocs
            )
        else:
            selected_data = self.get_selected_data(
                sessions=area_sessions, bin_width=bin_width, delay=delay,
                threshold=threshold, lag=lag, mVocs=mVocs
            )
        # if threshold is not None:
        #     selected_data = selected_data[
        #         (selected_data['normalizer']>=threshold)
        #     ]
        
        # selected_data = selected_data[
        #     (selected_data['tmax']==lag)
        # ]

        return selected_data[column]
    
    # def get_corr_for_area(
    #         self, neural_area='core',bin_width=20, delay=0,
    #         threshold=0.068, lag=None, normalized=True
    #     ):
    #     """Retrieves baseline correlations for all sig. sessions."""
    #     area_sessions = self.metadata.get_all_sessions(neural_area)

    #     corr_dist = self.get_correlations(
    #         sessions=area_sessions, bin_width=bin_width, delay=delay,
    #         threshold=threshold, lag=lag, normalized=normalized
    #     )
    #     return corr_dist




    # @staticmethod
    # def combine_and_ready(
    #         model_name, identifiers_list, output_id, normalizer_filename=None,
    #         output_identifier=None
    #     ):
    #     """Merges results for all identifiers, copies layer types and
    #     sets normalizer."""

    #     BaseCorrelations.merge_correlation_results(
    #             model_name=model_name,
    #             identifiers_list=identifiers_list,
    #             output_id=output_id,
    #             output_identifier=output_identifier
    #         )
        
    #     if output_identifier is None:
    #         output_identifier = identifiers_list[output_id]

        # print("Updating Normalizer...!")
        # res = model_name + '_' + output_identifier
        # corr_obj = STRFCorrelations(res, normalizer_filename=normalizer_filename)
        # # make sure 'test_cc_raw' and 'normalizer' columns are set...!
        # corr_obj.data['test_cc_raw'] = corr_obj.data['strf_corr']
        # corr_obj.set_normalizers()






























# from naplib.visualization import imSTRF


class Correlations(BaseCorrelations):
    def __init__(self, model_name=None) -> None:
        
        if model_name is None:
            model_name = 'wav2letter_modified_trained_all_bins'

        super().__init__(model_name)

        # self.model = model_name
        # filename = f'{model_name}_corr_results.csv'
        # self.corr_file_path = os.path.join(saved_corr_dir, filename)
        # self.metadata = NeuralMetaData()
        # self.norm_obj = Normalizer(normalizer_filename)


        # self.data = pd.read_csv(self.corr_file_path)
        # # check if 'test_cc_raw' not one of the columns
        # # this will be the case for STRF correlations
        # # in that case, copy 'strf_corr' to 'test_cc_raw'
        # if 'test_cc_raw' not in self.data.columns:
        #     self.data['test_cc_raw'] = self.data['strf_corr']
        #     print(f"'test_cc_raw added as a column..!")
        
        # if 'normalizer' not in self.data.columns:
        #     self.data.loc[:, 'normalizer'] = np.nan
        #     # self.set_normalizers()
        #     self.set_normalizers_using_bootsrap()
        #     # CorrelationUtils.copy_normalizer(model_name)
        #     # self.data = pd.read_csv(self.corr_file_path)


        if 'N_sents' not in self.data.columns:
            self.data['N_sents'] = np.ones(len(self.data.index))*500

        # self.data['normalized_test_cc'] = self.data['test_cc_raw']/(self.data['normalizer'].apply(np.sqrt))
        # self.sig_threshold = sig_threshold

        # loading STRF baseline
        third = None
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
        layer_types = ['conv', 'rnn', 'transformer', 'mix']
        self.fill_color = {}
        for layer, color in zip(layer_types, colors):
            self.fill_color[layer] = color

        # if 'normalizer_app' not in self.data.columns:
        #     # need to remove these bad channels...
        #     bad_sessions = [180725., 190607., 191212., 200226.]
        #     all_bad_ids = []
        #     for sess in bad_sessions:
        #         all_bad_ids.append(self.data[self.data['session'] == sess].index)
        #     all_bad_ids = np.concatenate(all_bad_ids)
        #     self.data.drop(all_bad_ids, inplace=True)

        #     self.set_normalizers_using_app()
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
            self, sessions=None, bin_width=20, delay=0, column='strf_corr', threshold=None,
            normalized=True):
        
        column='strf_corr'
        if normalized:
            column += '_normalized'

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
    # Moved to BaseCorrelation    
    # def get_filepath(self):
    #     """Returns abs path of corr result file."""
    #     return self.corr_file_path

    
    # def write_back(self):
    #     self.data.to_csv(self.corr_file_path, index=False)
    #     print(f"Saved at {self.corr_file_path}")
    # ##### TMPORARY function...to be removed 
    # def add_layer_types(self, other_type):
    #     self.data['layer_type'] = other_type
    #     self.write_back()

    # ---------------- methods using normalizer dist. using random pairs of trials ---------#
#moved_to_base
    # def set_normalizers_using_bootsrap(self):
    #     """Uses normalizer distribution computed using 100k runs 
    #     with random concatenation order of trials, and takes following steps:
    #         - adds a column 'normalizer' containig the new normalizer.
    #         - adds a column 'normalized_test_cc' containing corrected correlations.
    #     """
    #     bin_widths = self.data['bin_width'].unique()
    #     for bin_width in bin_widths:
    #         select_data = self.get_selected_data(bin_width=bin_width)
    #         sessions = select_data['session'].unique()
    #         delays = select_data['delay'].unique()
    #         for session in sessions:
    #             for delay in delays:
    #                 select_data = self.get_selected_data(
    #                     sessions=[session], bin_width=bin_width, delay=delay
    #                 )
    #                 channels = select_data['channel'].unique()

    #                 # reading the normalizer...
    #                 normalizers_dist = self.norm_obj.get_normalizer_for_session_random_pairs(
    #                     session=session, bin_width=bin_width 
    #                 )
    #                 normalizers_dist = np.median(normalizers_dist, axis=0)

    #                 for ch in channels:
    #                     ch = int(ch)
    #                     ch_normalizer = normalizers_dist[ch]
    #                     ids = select_data[select_data['channel']==ch].index
    #                     self.data.loc[ids, 'normalizer'] = ch_normalizer 

    #     self.data['normalized_test_cc'] = self.data['test_cc_raw']/(self.data['normalizer'].apply(np.sqrt))
    #     print(f"Normalizers updated using normalizer (random pairs) dist , writing back now...")
    #     self.write_back()


    # ---------------- methods using normalizer dist. using all-possible-pairs (app) ---------#
    # moved_to_base
    # def set_normalizers_using_app(self):
    #     """Uses normalizer distribution computed using all-possible-pairs (app),
    #     and takes following steps:
    #         - adds a column 'normalizer_app' containig the new normalizer.
    #         - adds a column 'corr_normalized_app' containing corrected correlations.
    #     """
    #     bin_widths = self.data['bin_width'].unique()
    #     for bin_width in bin_widths:
    #         select_data = self.get_selected_data(bin_width=bin_width)
    #         sessions = select_data['session'].unique()
    #         delays = select_data['delay'].unique()
    #         for session in sessions:
    #             for delay in delays:
    #                 select_data = self.get_selected_data(
    #                     sessions=[session], bin_width=bin_width, delay=delay
    #                 )
    #                 channels = select_data['channel'].unique()

    #                 # reading the normalizer...
    #                 normalizers_dist = self.norm_obj.get_normalizer_for_session_app(
    #                     session=session, bin_width=bin_width 
    #                 )
    #                 normalizers_dist = np.mean(normalizers_dist, axis=0)

    #                 for ch in channels:
    #                     ch = int(ch)
    #                     ch_normalizer = normalizers_dist[ch]
    #                     ids = select_data[select_data['channel']==ch].index
    #                     self.data.loc[ids, 'normalizer_app'] = ch_normalizer 

    #     self.data['corr_normalized_app'] = self.data['test_cc_raw']/(self.data['normalizer_app'].apply(np.sqrt))
    #     print(f"Normalizers updated using normalizer (app) dist , writing back now...")
    #     self.write_back()

    def get_significant_data_using_statistical_inclusion(
                self, bin_width, sessions: list=None, delay=None,
                N_sents=499, layer=None, inclusion_p_threshold=0.01,
                use_poisson_null=True, mVocs=False
        ):
        """Retrieves selected data based on provided arguments. 
        If an argument if 'None', no filter is applied on that column.
        Only retrieves data for signigicant neurons (selected based on 
        t-test by normalizer object.)

        Args:
            bin_width: int = bin_width in ms.
            sessions: list of ints = list of sessions IDs to get data for,
                if no session is provided, all significant sessions are retrieved.
            delay: int = delay in ms.

        
        Returns:
            pandas DataFrame
        """
        logger.info(f"Retrieving significant data using statistical inclusion and ", end='')
        select_data = self.data
        select_data = select_data[select_data['bin_width']==float(bin_width)]
        
        if delay is not None:
            select_data = select_data[select_data['delay']==float(delay)]
        
        if N_sents is not None:
            select_data = select_data[select_data['N_sents']>=float(N_sents)]
        
        # if threshold is not None:
        #     select_data = select_data[select_data['normalizer']>=float(threshold)]

        if layer is not None:
            select_data = select_data[select_data['layer']==float(layer)]

        ## select significant sessions....
        ## get sig. sessions and channels
        ## filter the channels also..
        if use_poisson_null:
            logger.info(f"Poisson Null...")
            sig_sess_n_chs = self.norm_obj.get_significant_sessions_and_channels_using_poisson_null(
                bin_width=bin_width, p_threshold = inclusion_p_threshold, mVocs=mVocs
                )
        else:
            logger.info(f"Random shifts Null...")
            sig_sess_n_chs = self.norm_obj.get_significant_sessions_and_channels_using_shifts_null(
                bin_width=bin_width, p_threshold = inclusion_p_threshold, mVocs=mVocs
                )

        if sessions is None:
            sessions = list(sig_sess_n_chs.keys())

        filtered_data = []
        # sessions is a list...
        for session in sessions:
            session = str(int(float(session)))
            if session in sig_sess_n_chs.keys():
                session_data = select_data[(select_data['session']==float(session))]
                for ch in sig_sess_n_chs[session]:
                    filtered_data.append(
                        session_data[session_data['channel']==float(ch)]
                    )

        filtered_data = pd.concat(filtered_data)
        return filtered_data










    # ---------------- Methods defined before app normalizers ---------#

    def get_significant_sessions(self, threshold = None, bin_width=50, neural_area=None):
        """Returns sessions with corr scores above significant threshold for at least 1 channel"""
        sessions = self.metadata.get_all_sessions(neural_area)
        if threshold is None:
            threshold = self.get_normalizer_threshold(bin_width=bin_width, poisson_normalizer=True)
        sig_data = self.get_selected_data(
            sessions=sessions, bin_width=bin_width, threshold=threshold
            )
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
                self, sessions: list=None, bin_width=None, delay=None, threshold=None,
                N_sents=None, layer=None, channel=None, mVocs=False
        ):
        """Retrieves selected data based on provided arguments. 
        If an argument if 'None', no filter is applied on that column.

        Args:
            sessions: list of ints = list of sessions IDs to get data for.
            bin_width: int = bin_width in ms.
            delay: int = delay in ms.
            threshold: float = correlation threshold for repeatability
            mVocs: bool = If True, use 'mVocs_normalizer' for significance
        
        Returns:
            pandas DataFrame
        """
        # print(f"get_selected_data() method inside Correlations....")
        select_data = self.data
        if bin_width is not None:
            select_data = select_data[select_data['bin_width']==float(bin_width)]
        
        if delay is not None:
            select_data = select_data[select_data['delay']==float(delay)]
        
        if N_sents is not None:
            select_data = select_data[select_data['N_sents']>=float(N_sents)]
        
        if threshold is not None:
            select_data = self.get_highly_tuned_channels(select_data, threshold=threshold, mVocs=mVocs)
            # norm_column = 'normalizer'
            # if mVocs:
            #     norm_column = 'mVocs_'+norm_column
            # print(f"Applying threshold: {threshold:.3f} on column: '{norm_column}'...")
            # select_data = select_data[select_data[norm_column]>=float(threshold)]

        if layer is not None:
            select_data = select_data[select_data['layer']==float(layer)]

        if channel is not None:
            select_data = select_data[select_data['channel']==float(channel)]

        if sessions is not None:
            session_data = []
            # sessions is a list...
            for session in sessions:
                session_data.append(select_data[
                        (select_data['session']==float(session))
                    ])
            select_data = pd.concat(session_data)
        
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
            logger.info(f"Baseline median: {np.median(baseline_corr):.3f}")
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
    
# ------------------  Retrieve data for analysis ----------------#

# moved_to_base
    # def get_normalizer_threshold(self, bin_width=20, poisson_normalizer=True):
    #     """Retrieves significance threshold from the NULL distribution
    #     of normalizer.
    #     """
    #     threshold_dict = io.read_normalizer_threshold(
    #         bin_width=bin_width, poisson_normalizer=poisson_normalizer
    #         )
    #     if threshold_dict is None or bin_width not in threshold_dict.keys():
    #         if poisson_normalizer:
    #             null_dist = self.norm_obj.get_normalizer_null_dist_using_poisson(bin_width=bin_width)
    #             threshold = np.percentile(null_dist, 95)    
    #             # threshold = self.norm_obj.compute_normalizer_threshold_using_poisson(bin_width=bin_width)[0]
    #         else:
    #             threshold = self.norm_obj.compute_normalizer_threshold(bin_width=bin_width)[0]

    #         io.write_normalizer_threshold(
    #             bin_width=bin_width, poisson_normalizer=poisson_normalizer,
    #             thresholds=threshold
    #         )
    #         return threshold
    #     else:
    #         return threshold_dict[bin_width]


    def get_corr_all_bin_widths_for_layer(
            self, neural_area=None, layer=6, delay=0, N_sents=499, 
            poisson_normalizer=True, normalized=True
        ):
        """Retrieves correlations for all layers, but ONLY the specified bin_width."""
        assert neural_area in ['core', 'belt', 'all'], logger.info(f"Unknown neural area '{neural_area}' specified.")
        if normalized:
            column = 'normalized_test_cc'
        else:
            column = 'test_cc_raw'

        area_sessions = self.metadata.get_all_sessions(neural_area)

        dist_spread = {}   
        bin_widths = np.sort(self.data['bin_width'].unique())
        for bin_width in bin_widths:
            bw_threshold = self.get_normalizer_threshold(
                bin_width=bin_width, poisson_normalizer=poisson_normalizer
            )
            # if poisson_normalizer:
            #     bw_threshold = self.norm_obj.compute_normalizer_threshold_using_poisson(bin_width=bin_width)[0]
            # else:
            #     bw_threshold = self.norm_obj.compute_normalizer_threshold(bin_width=bin_width)[0]
            select_data = self.get_selected_data(
                sessions=area_sessions,
                bin_width=bin_width, 
                layer=layer,
                delay=delay,
                threshold=bw_threshold,
                N_sents=N_sents
            )
            # ids = select_data[select_data['bin_width']==bin_width].index
            dist_spread[int(bin_width)] = np.array(select_data[column]).squeeze()


        # select_data = self.get_selected_data(
        #     sessions=area_sessions, 
        #     layer=layer,
        #     delay=delay,
        #     threshold=threshold,
        #     N_sents=N_sents
        # )

        # dist_spread = {}   
        # bin_widths = select_data['bin_width'].unique()
        # for bin_width in bin_widths:
        #     ids = select_data[select_data['bin_width']==bin_width].index
        #     dist_spread[int(bin_width)] = np.array(select_data.loc[ids, column]).squeeze()
        # sort the result based on keys...
        dist_spread = dict(sorted(dist_spread.items(), key=lambda item: item[0]))
        return dist_spread
    
# ------------------  corr distributions of all layers  ----------------#
    
    def get_corr_all_layers_for_bin_width(self, neural_area=None, bin_width=20, delay=0, 
        N_sents=499, threshold = 0.068, normalized=True, column=None, mVocs=False, 
        use_stat_inclusion=False, inclusion_p_threshold=0.01, use_poisson_null=True):
        """Retrieves correlations for all layers, but ONLY the specified bin_width.
        
        Args:
            use_stat_inclusion: bool = If True, select significant sessions/channels 
                using t-test instead of thresholding method.
        """
        area_choices = self.metadata.get_area_choices()
        # assert neural_area in ['core', 'belt', 'parabelt', 'all'], print(f"Unknown neural area '{neural_area}' specified.")
        assert neural_area in area_choices, logger.info(f"Unknown neural area '{neural_area}' specified.")
        if column is None:
            if normalized:
                # Deprecated...
                # if use_stat_inclusion:
                    # column = 'corr_normalized_app'
                # else:
                column = 'normalized_test_cc'
            else:
                column = 'test_cc_raw'
            if mVocs:
                column = 'mVocs_'+column

        logger.info(f"Extracting column: {column}")

        area_sessions = self.metadata.get_all_sessions(neural_area)
        if use_stat_inclusion:
            select_data = self.get_significant_data_using_statistical_inclusion(
                bin_width=bin_width, sessions=area_sessions, delay=delay,
                inclusion_p_threshold=inclusion_p_threshold,
                use_poisson_null=use_poisson_null, mVocs=mVocs
            )
        else:
            select_data = self.get_selected_data(
                sessions=area_sessions, 
                bin_width=bin_width,
                delay=delay,
                threshold=threshold,
                N_sents=N_sents,
                mVocs=mVocs,
            )

        layer_spread = {}   
        layer_IDs = select_data['layer'].unique()
        for layer in layer_IDs:
            ids = select_data[select_data['layer']==layer].index
            layer_spread[int(layer)] = np.array(select_data.loc[ids, column]).squeeze()
        # sort the result based on keys...
        layer_spread = dict(sorted(layer_spread.items(), key=lambda item: item[0]))
        return layer_spread


    
    

# ------------------  corr KDE for all layers  ----------------#

    def get_KDE_all_layers_for_bin_width(
            self, neural_area='core', bin_width=20, delay=0,
            snippet_width = 100, num_hist_bins = 1000, 
            normalized=True, poisson_normalizer=True
        ):
    


        bw_threshold = self.get_normalizer_threshold(
                        bin_width=bin_width, poisson_normalizer=poisson_normalizer
                        )

        data_dist = self.get_corr_all_layers_for_bin_width(
                neural_area=neural_area, bin_width=bin_width,
                delay=delay, threshold=bw_threshold,
                normalized=normalized
            )

        points = np.linspace(0,1,num_hist_bins)
        density = True
        vect_ones = np.ones(snippet_width)[None,:]
        num_layers = len(data_dist.keys())
        hist_combined = np.ones((num_hist_bins, num_layers*snippet_width))
        for layer_id in data_dist.keys():
            data = data_dist[layer_id]
            kde_obj = scp.stats.gaussian_kde(data)
        
            layer_hist = kde_obj(points)
            layer_hist /= np.sum(layer_hist)

            # replicating layer_hist 'snippet_width' times...
            hist_snippet = layer_hist[:, None] @ vect_ones
            hist_combined[:, layer_id*snippet_width:(layer_id+1)*snippet_width] = hist_snippet
    
        ytick_labels = np.linspace(0,1,5)
        yticks = ytick_labels*num_hist_bins

        x_tick_labels = np.array(list(data_dist.keys()))
        x_ticks = (x_tick_labels)*snippet_width + snippet_width//2

        return hist_combined, (x_ticks, x_tick_labels), (yticks, ytick_labels)
    
    def get_KDE_for_baseline(
            self, area, bin_width, delay=0,
            normalized=True, poisson_normalizer = True 
        ):
        """Get KDE for baseline data."""
        bw_threshold = self.get_normalizer_threshold(
                bin_width=bin_width, poisson_normalizer=poisson_normalizer
                )

        # plot baseline...
        area_sessions = self.metadata.get_all_sessions(area)
        baseline_dist = self.get_baseline_corr_session(
            sessions= area_sessions,bin_width=bin_width, delay=delay,
                    threshold=bw_threshold, normalized=normalized)

        data_dist = {0: baseline_dist}

        return self.get_KDE_from_dist(data_dist)


    def get_KDE_from_dist(
            self, data_dist, 
            snippet_width = 100, num_hist_bins = 1000, 
        ):
        """Coverts dist of data points into KDE distribution, 
        ready to be plotted.
        """
        points = np.linspace(0,1,num_hist_bins)
        density = True
        vect_ones = np.ones(snippet_width)[None,:]
        num_layers = len(data_dist.keys())
        hist_combined = np.ones((num_hist_bins, num_layers*snippet_width))
        for layer_id in data_dist.keys():
            data = data_dist[layer_id]
            kde_obj = scp.stats.gaussian_kde(data)
        
            layer_hist = kde_obj(points)
            layer_hist /= np.sum(layer_hist)

            # replicating layer_hist 'snippet_width' times...
            hist_snippet = layer_hist[:, None] @ vect_ones
            hist_combined[:, layer_id*snippet_width:(layer_id+1)*snippet_width] = hist_snippet
    
        ytick_labels = np.linspace(0,1,5)
        yticks = ytick_labels*num_hist_bins

        x_tick_labels = np.array(list(data_dist.keys()))
        x_ticks = (x_tick_labels)*snippet_width + snippet_width//2

        return hist_combined, (x_ticks, x_tick_labels), (yticks, ytick_labels)



    ## ------------ analysis at all bin widths -----------------------------##

    def get_significant_session_and_channels_at_all_bin_width(
            self, poisson_normalizer = True,
        ):
        """Retrieves significant sessions and channels at each bin-width using normalizer
        threshold (different for each bin width), and returns are super-set 
        of sessions-channels as a dict.

        Args:
            poisson_normalizer: bool = If True, uses poisson normalizer.

        Returns:
            dict: significant sessions as keys and significant channels against each session as values.
        """
        sig_sessions = {}
        all_session_channels = {}
        bin_widths = np.sort(self.data['bin_width'].unique())
        for bin_width in bin_widths:
            bw_threshold = self.get_normalizer_threshold(bin_width=bin_width, poisson_normalizer=poisson_normalizer)
            sig_sessions[bin_width] = self.get_significant_sessions(threshold=bw_threshold, bin_width=bin_width)

            for session in sig_sessions[bin_width]:
                channels = self.get_selected_data(
                    sessions=[session], bin_width=bin_width, delay=0, threshold=bw_threshold
                )['channel']

                if session in all_session_channels.keys():
                    all_session_channels[session].extend(list(np.unique(channels)))
                else:
                    all_session_channels[session] = list(np.unique(channels))

        for sess, channels in all_session_channels.items():
            all_session_channels[sess] = np.unique(channels)

        return all_session_channels
    

    def get_corr_super_set_all_layers_for_bin_width(
            self, sig_session_channel_dict, bin_width=20, delay=0, 
            normalized=True, column=None
        ):
        """Retrieves corr dist using 'superset' (all session-channels significant at any bin-width),
        and zeroing out the values for extra entries i.e. the ones that would not qualify based on 
        normalizer threshold.
        
        """
        if column is None:
            if normalized:
                column = 'normalized_test_cc'
            else:
                column = 'test_cc_raw'
        else:
            logger.info(f"Extracting column: {column}")

        bin_width_data = []
        sessions = np.sort(list(sig_session_channel_dict.keys()))
        for sess in sessions:
            select_data = self.get_selected_data(
                sessions=[sess], delay=delay, bin_width=bin_width
            )

            session_data = []
            # channels is a list...
            channels = np.sort(sig_session_channel_dict[sess])
            for channel in channels:
                session_data.append(select_data[
                        (select_data['channel']==float(channel))
                    ])
            session_data = pd.concat(session_data)
            bin_width_data.append(session_data)
        bin_width_data = pd.concat(bin_width_data)

        # replace the entries with insignificant normalizer with zero...
        bw_threshold = self.get_normalizer_threshold(bin_width=bin_width, poisson_normalizer=True)
        insignificant_ids = bin_width_data[bin_width_data['normalizer'] < float(bw_threshold)].index
        bin_width_data.loc[insignificant_ids, column] = 0

        # get all layer distributions...
        layer_spread = {}   
        layer_IDs = bin_width_data['layer'].unique()
        for layer in layer_IDs:
            ids = bin_width_data[bin_width_data['layer']==layer].index
            layer_spread[int(layer)] = np.array(bin_width_data.loc[ids, column]).squeeze()
        # sort the result based on keys...
        layer_spread = dict(sorted(layer_spread.items(), key=lambda item: item[0]))
        return layer_spread



    
    def get_layer_dist_with_peak_median_using_super_set(
            self, sig_session_channel_dict,
            bin_width, delay=0, normalized=True, column=None,
        ):
        """Returns the corr distribution, at specific bin width,
        for the layer with peak median. Takes in super-set of sessions-channels.
        """
        layer_spread = self.get_corr_super_set_all_layers_for_bin_width(
            sig_session_channel_dict, bin_width=bin_width, normalized=normalized,
            column=column, delay=delay
            )
        
        layer_medians = {np.median(v):k for k,v in layer_spread.items()}
        peak_median = max(layer_medians)
        peak_layer = layer_medians[peak_median]
        logger.info(f"At bin_width: {bin_width}, layer with peak median is: {peak_layer}")
        return layer_spread[peak_layer]



    def get_layer_dist_with_peak_median(
            self, 
            bin_width,
            threshold=None,
            neural_area='all',
            delay=0,
            normalized=True, 
            mVocs=False,
            poisson_normalizer=True,
            threshold_percentile=None,
            norm_bin_width=None,
            layer_id=None,
        ):
        """Returns the corr distribution, at specific bin width,
        for the layer with peak median.

        Args:
            layer_id: int = If specified, returns dist for layer_id,
                    else returns the for layer with peak median.
        """
        if norm_bin_width is None:
            norm_bin_width = bin_width
        if threshold is None:
            threshold = self.get_normalizer_threshold(
                    bin_width=norm_bin_width, 
                    mVocs=mVocs,
                    poisson_normalizer=poisson_normalizer,
                    threshold_percentile=threshold_percentile, 
                    )
        corr_dict = self.get_corr_all_layers_for_bin_width(
                neural_area=neural_area, bin_width=bin_width,
                delay=delay, threshold=threshold,
                normalized=normalized, mVocs=mVocs
            )

        if layer_id is None:
            #pick the peak layer..
            layer_medians = {np.median(v):k for k,v in corr_dict.items()}
            peak_median = max(layer_medians)
            peak_layer = layer_medians[peak_median]
            logger.info(f"At bin_width: {bin_width}, layer with peak median is: {peak_layer}")
            layer_id = peak_layer
        else:
            logger.info(f"Getting dist for layer_id={layer_id}")
        logger.info(f"Number of sig. neurons = {corr_dict[layer_id].shape[0]}")
        return corr_dict[layer_id]
    

    def get_baseline_corr_for_area(
            self, neural_area='core',bin_width=20, delay=0,
            threshold=0.068, normalized=True
        ):
        """Retrieves baseline correlations for all sig. sessions."""
        area_sessions = self.metadata.get_all_sessions(neural_area)

        corr_dist = self.get_baseline_corr_session(
            sessions=area_sessions, bin_width=bin_width, delay=delay,
            threshold=threshold, normalized=normalized
        )
        return corr_dist
    
    def get_architecture_specific_layer_ids(self):
        """
        Returns list of layer ids for each architecture 
        type e.g.{'conv': [0,1],
                'rnn': [2,3,4,5]}
        """
        data_series = self.data.groupby('layer_type')['layer'].unique()
        layer_arch_types = data_series.index
        arch_specific_layer_ids = {}
        for layer in layer_arch_types:
            arch_specific_layer_ids[layer] =  data_series[layer]
        return arch_specific_layer_ids
        
    # moved_to_base
# # ------------------  copy normalizer for all bin-widths ----------------#
#     def set_normalizers(self, bin_widths: list=None):
#         """Set normalizers (repeatability correlations) from the saved results.
#         """
#         if bin_widths is None:
#             bin_widths = self.data['bin_width'].unique()
#         for bin_width in bin_widths:
#             select_data = self.get_selected_data(bin_width=bin_width)
#             sessions = select_data['session'].unique()
#             delays = select_data['delay'].unique()
#             for session in sessions:
#                 for delay in delays:
#                     select_data = self.get_selected_data(
#                         sessions=[session], bin_width=bin_width, delay=delay
#                     )
#                     channels = select_data['channel'].unique()
#                     selected_normalizers = self.norm_obj.get_normalizer_for_session(
#                         session=session, bin_width=bin_width, delay=delay
#                         )
#                     for ch in channels:
#                         ids = select_data[select_data['channel']==ch].index
#                         ch_normalizer = selected_normalizers[
#                                 selected_normalizers['channel']==ch
#                             ]['normalizer'].head(1).item()
#                         self.data.loc[ids, 'normalizer'] = ch_normalizer #/0.82   # adjusting for context..

#         self.data['normalized_test_cc'] = self.data['test_cc_raw']/(self.data['normalizer'].apply(np.sqrt))
#         print(f"Normalizers updated FOR CONTEXT AS WELL, writing back now...")
#         self.write_back()


# ------------------    combine resutls, add layer types and normalizer     ----------------#



    # @staticmethod
    # def combine_and_ready(
    #         model_name, identifiers_list, output_id, normalizer_filename=None,
    #         output_identifier=None
    #     ):
    #     """Merges results for all identifiers, copies layer types and
    #     sets normalizer."""

    #     BaseCorrelations.merge_correlation_results(
    #             model_name=model_name,
    #             identifiers_list=identifiers_list,
    #             output_id=output_id,
    #             output_identifier=output_identifier
    #         )
        
    #     if output_identifier is None:
    #         output_identifier = identifiers_list[output_id]

    #     # identifier = identifiers_list[output_id]
    #     Correlations.add_layer_types(
    #         model_name, output_identifier
    #     )

        # setting the normalizer..
        # res = model_name + '_' + output_identifier
        # corr_obj = Correlations(res, normalizer_filename=normalizer_filename)

        # corr_obj.set_normalizers()
        # corr_obj.set_normalizers_using_bootsrap()

    #################################################
    ### June 23, 2024: Moved to BaseCorrelations ####
    #################################################

    # @staticmethod
    # def merge_correlation_results(model_name, file_identifiers, idx, output_identifier=None):
    #     """
    #     Args:

    #         model_name: Name of the pre-trained network
    #         file_identifiers: List of filename identifiers 
    #         idx:    id of the file identifier to use for saving the merged results
    #         output_identifier: if output_identifier is given, it takes preference..
    #     """
    #     # results_dir = '/depot/jgmakin/data/auditory_cortex/correlation_results/cross_validated_correlations'

    #     corr_dfs = []
    #     for identifier in file_identifiers:
    #         filename = f"{model_name}_{identifier}_corr_results.csv"
    #         file_path = os.path.join(saved_corr_dir, filename)

    #         corr_dfs.append(pd.read_csv(file_path))

    #     # save the merged results at the very first filename...
    #     if output_identifier is None:
    #         output_identifier = file_identifiers[idx]    
    #     filename = f"{model_name}_{output_identifier}_corr_results.csv"
    #     file_path = os.path.join(saved_corr_dir, filename)

    #     data = pd.concat(corr_dfs)
    #     data.to_csv(file_path, index=False)
    #     print(f"Output saved at: \n {file_path}")

    #     # once all the files have been merged, remove the files..
    #     for identifier in file_identifiers:
    #         if identifier != output_identifier:
    #             filename = f"{model_name}_{identifier}_corr_results.csv"
    #             file_path = os.path.join(saved_corr_dir, filename)
    #             # remove the file
    #             os.remove(file_path)


    @staticmethod
    def add_layer_types(model_name, results_identifer):

        # reading layer_types from aux config...
        layer_types = {}
        config_file = os.path.join(aux_dir, f"{model_name}_config.yml")
        with open(config_file, 'r') as f:
            config = yaml.load(f, yaml.FullLoader)

        # config['layers']
        for layer_config in config['layers']:
            layer_types[layer_config['layer_id']] = layer_config['layer_type']

        # reading results directory...
        if results_identifer != '':
            model = f'{model_name}_{results_identifer}'
        else:
            model = model_name 
        filename = f"{model}_corr_results.csv"
        file_path = os.path.join(saved_corr_dir, filename)
        data = pd.read_csv(file_path)
        logger.info(f"reading from {file_path}")

        # remove 'Unnamed' columns
        data = data.loc[:, ~data.columns.str.contains('Unnamed')]

        # add 'layer_type' as a column
        for layer, type in layer_types.items():
            ids = data[data['layer']==layer].index
            data.loc[ids, 'layer_type'] = type

        logger.info("Writing back...!")
        data.to_csv(file_path, index=False)



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




    


