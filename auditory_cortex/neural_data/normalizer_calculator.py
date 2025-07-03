import os
import time
import numpy as np
import pandas as pd
import scipy

# from .deprecated.dataset import NeuralData
# from .deprecated.neural_meta_data import NeuralMetaData
from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata, BaseDataset
from auditory_cortex import results_dir
import auditory_cortex.io_utils.io as io

import logging
logger = logging.getLogger(__name__)

class NormalizerCalculator:
    """Computes the inter-trial correlations for stimuli having multilple trials.
    These will be used as normalizer for regression correlations and for identifying channels
    (and sessions) with good enough SNR. In summary, this class does the following:
    - Compute the normalizer distribution.
    - Compute the null distribution.
    - Save these distributions to disk.
    """

    def __init__(self, dataset_name):
        
        self.dataset_name = dataset_name
        self.metadata = create_neural_metadata(dataset_name)

    ######################################################################
    ################## methods simplified: 06-24-25: starts here...
    ######################################################################

    def get_repeated_spikes(self, session, bin_width=50, mVocs=False):
        """Reads spikes for repeated stimuli and returns a dictionary
        
        Args:
            bin_width: int = bin width in ms
            mVocs: bool = If True, mVocs trials are considered.
        Returns:
            dict: {stim_id: {ch: array}} = where ndarray is of shape (num_repeats, seq_len) 
        """
        dataset = create_neural_dataset(self.dataset_name, session)
        spikes = dataset.extract_spikes(
            bin_width=bin_width, delay=0, repeated=True, mVocs=mVocs
            )
        return spikes
    

    def get_testing_stim_duration(self, mVocs=False):
        """Returns total duration of testing stimuli"""
        stim_duration = self.metadata.total_stimuli_duration(mVocs)
        return stim_duration['repeated']

    def get_test_set_ids(self, percent_duration=None, mVocs=False):
        """Returns random choice of stimulus ids, for the desired fraction of total 
        duration of test set as specified by percent_duration.
        
        Args:
            percent_duration: float = Percentage of total duration of test set to consider.
                If None or >= 100, returns all stimulus ids.
            mVocs: bool = If True, mVocs trials are considered otherwise TIMIT
        
        Returns:
            list: List of stimulus ids to consider for testing.
        """
        stim_ids, stim_duration = self.metadata.sample_stim_ids_by_duration(
            percent_duration, repeated=True, mVocs=mVocs
            )
        logger.info(f"Total duration={stim_duration:.2f} sec")
        return stim_ids

    def _compute_inter_trial_corr_dists(
            self, session, bin_width=50, num_itr=1000000, mVocs=False
            ):
        """Compute dist. of normalizer (true & null) for correlations (repeatability of neural
        spikes). The only difference between true and null distribution is whether one sequence
        is circularly shifted w.r.t. the other.
        
        Args:
            session: str = session id
            bin_width: int = bin width in ms
            num_itr: int = number of iterations for distribution
            mVocs: bool = If True, mVocs trials are considered otherwise TIMIT

        Returns:
            norm_dists (dict): {channel: (num_itr,)} True distribution of inter-trial correlations
            null_dists (dict): {channel: (num_itr,)} Null distribution of inter-trial correlations
        """
        repeated_spikes = self.get_repeated_spikes(
            session=session, bin_width=bin_width, mVocs=mVocs
            )
        norm_dist, null_dist = self.inter_trial_corr_using_random_pairing(
            repeated_spikes, num_itr=num_itr
            )
        return norm_dist, null_dist

    def get_inter_trial_corr_dists_for_session(
            self, session, bin_width=50, mVocs=False, num_itr=10000, **kwargs
            ):
        """Retrieves the distribution of normalizer for all the channels 
        of the specified session, at specified bin_width and delay.

        Args:
            session: str = session id
            bin_width: int = bin width in ms
            mVocs: bool = If True, mVocs trials are considered otherwise TIMIT
            num_itr: int = number of iterations for distribution

            kwargs:
                - force_redo: bool = If True, recomputes the distribution even if it exists on disk, Default=False
                    
        Returns:
            norm_dist (dict): {channel: (num_itr,)} True distribution of inter-trial correlations
            null_dist (dict): {channel: (num_itr,)} Null distribution of inter-trial correlations
        """
        force_redo = kwargs.get('force_redo', False)
        session = str(int(float(session)))    # to make sure session is in str format.
        bin_width = int(bin_width)
        dataset_name = self.dataset_name
        logger.info(f"Getting normalizer dist. for sess-{session}, bw-{bin_width}, mVocs={mVocs}")

        if dataset_name=='ucsf' and mVocs:
            # handling the excluded sessions for mVocs...
            if session == '190726' or session == '200726':
                raise ValueError(
                    f"Session {session} is not available for mVocs. Please check the dataset."
                    )
            
        if not force_redo:
            norm_dist, null_dist = io.read_inter_trial_corr_dists(
                session, bin_width, mVocs=mVocs, dataset_name=dataset_name
            )
        if force_redo or (norm_dist is None) or (null_dist is None):
            norm_dist, null_dist = self._compute_inter_trial_corr_dists(
                session, bin_width=bin_width, mVocs=mVocs, num_itr=num_itr
            )
            io.write_inter_trial_corr_dists(
                norm_dist, null_dist,
                session, bin_width, mVocs=mVocs, dataset_name=dataset_name
            )
        return norm_dist, null_dist
    
    def get_bootstrap_distributions(
        self, session, percent_durations:list, epoch_ids:list, 
        bin_width=50, mVocs=False
        ):
        """Retrieves distributions of normalizer for bootstrap analysis,
        for the specified session.

        Args:
            session: str = session id
            percent_durations: list = List of percent durations to consider for bootstrapping.
            epoch_ids: list = List of epoch indices for bootstrapping. If None, defaults to [1].
            bin_width: int = bin width in ms
            mVocs: bool = If True, mVocs trials are considered otherwise TIMIT

        Returns:
            dict of dists: { (percent_dur, num_trials): (norm_dists, null_dists) }
            where norm_dists and null_dists are dictionaries of the form
            {ch: dist} where dist is a numpy.ndarray of shape (num_epochs, num_itr)
        """
        num_repeats = self.metadata.num_repeats_for_sess(session)
        num_trials_list = np.arange(2, num_repeats+1)
        bootstrap_dists = {}
        for percent_dur in percent_durations:
            for num_trials in num_trials_list:
                norm_dist_all_epochs = {}
                null_dist_all_epochs = {}
                for epoch in epoch_ids:
                    norm_dist, null_dist = io.read_inter_trial_corr_dists(
                        session, bin_width, mVocs=mVocs, dataset_name=self.dataset_name,
                        bootstrap=True, epoch=epoch, percent_dur=percent_dur, num_trial=num_trials
                    )
                    norm_dist_all_epochs[epoch] = norm_dist
                    null_dist_all_epochs[epoch] = null_dist
                
                ch_list = norm_dist_all_epochs[epoch].keys()
                all_norm_dists = {
                    ch: np.stack([norm_dist_all_epochs[epoch][ch] for epoch in epoch_ids])
                    for ch in ch_list
                }
                all_null_dists = {
                    ch: np.stack([null_dist_all_epochs[epoch][ch] for epoch in epoch_ids])
                    for ch in ch_list
                }
                bootstrap_dists[(percent_dur, num_trials)] = (all_norm_dists, all_null_dists)
        return bootstrap_dists
    
    def save_bootstrapped_distributions(
        self, session, percent_durations, epoch_ids=None, bin_width=50, num_itr=1000, mVocs=False
        ):
        """Computes and saves the bootstrapped distributions of inter-trial correlations
        for different setting of percent durations and number of repeats.

        Args:
            session: str = session id
            percent_durations: list = List of percent durations to consider for bootstrapping.
            epoch_ids: list = List of epoch indices for bootstrapping. If None, defaults to [1].
            bin_width: int = bin width in ms
            num_itr: int = number of iterations for distribution at each setting.
            mVocs: bool = If True, mVocs trials are considered otherwise TIMIT
        """
        repeated_spikes = self.get_repeated_spikes(
            session=session, bin_width=bin_width, mVocs=mVocs
            )
        
        stim_ids = list(repeated_spikes.keys())
        channel_ids = list(repeated_spikes[stim_ids[0]].keys())
        num_repeats = repeated_spikes[stim_ids[0]][channel_ids[0]].shape[0]
        if epoch_ids is None:
            epoch_ids = [0]
        num_trials_list = np.arange(2, num_repeats+1)
        for epoch in epoch_ids:
            for percent_dur in percent_durations:
                for num_trials in num_trials_list:
                    stim_ids = self.get_test_set_ids(percent_dur, mVocs=mVocs)
                    norm_dist, null_dist = self.inter_trial_corr_using_random_pairing(
                        repeated_spikes, num_itr=num_itr, stim_ids=stim_ids,  num_trials=num_trials
                        )
                    io.write_inter_trial_corr_dists(
                        norm_dist, null_dist, 
                        session, bin_width, mVocs=mVocs, dataset_name=self.dataset_name,
                        bootstrap=True, epoch=epoch, percent_dur=percent_dur, num_trial=num_trials
                    )


    # @staticmethod
    # def inter_trial_corr_for_bootstrap_analysis(
    #     sent_wise_repeated_spikes, stim_ids=None, n=10000, num_trials=3, 
    #     ):

    #     if stim_ids is None:
    #         stim_ids = list(sent_wise_repeated_spikes.keys())
    #     # num_channels = next(iter(sent_wise_repeated_spikes.values())).shape[-1]
    #     max_num_trials, _, num_channels = sent_wise_repeated_spikes[stim_ids[0]].shape
    #     trials_corr = np.zeros((n, num_channels))
    #     # if mVocs:
    #     # 	max_num_trials = 15
            
    #     # else:
    #     # 	max_num_trials = 11
    #     assert num_trials <= max_num_trials and num_trials >= 2, "num_trials must be between 1 and {}".format(max_num_trials)
    #     # trial_ids = np.arange(max_num_trials)
    #     # trial_ids = np.random.choice(trial_ids, size=num_trials, replace=False)
    #     trial_ids = np.random.choice(max_num_trials, size=num_trials, replace=True)

    #     for t in range(n):
    #         long_seq1 = []
    #         long_seq2 = []

    #         for stim_id in stim_ids:
    #             spikes = sent_wise_repeated_spikes[stim_id]
    #             tr1, tr2 = np.random.choice(trial_ids, size=2, replace=False)   
    #             # replace=False is important here, because, we wouldn't want to 
    #             # compute correlation between identical sequence. 

    #             long_seq1.append(spikes[tr1])
    #             long_seq2.append(spikes[tr2])
    #         long_seq1 = np.concatenate(long_seq1, axis=0)
    #         long_seq2 = np.concatenate(long_seq2, axis=0)

    #         for ch in range(num_channels):
    #             corr_ch = np.corrcoef(
    #                 long_seq1[...,ch].squeeze(), long_seq2[...,ch].squeeze()
    #                 )[0,1]
    #             # nan might result from entires sequence being zero, 
    #             # penalize that by setting corr equal to zero.
    #             if np.isnan(corr_ch):
    #                 corr_ch = 0
    #             trials_corr[t, ch] = corr_ch
    #     return trials_corr
   

            
    @staticmethod
    def inter_trial_corr_using_random_pairing(
        repeated_spikes, num_itr=100000, stim_ids=None, num_trials=None
        ):
        """Compute distribution of inter-trials correlations, using bootstrapping.
        At each iteration randomly selects trial pair for each sentence. Assigns one
        trial to first long sequence and second trial to second long sequence. 
        Computes both normalizer and null distribution of inter-trial correlations.

        Args: 
            repeated_spikes dict(stim: ndarray): {stim: {channel: (repeats, samples/time)}}
            num_itr (int): number of iterations
            stim_ids: list of str = List of stimulus ids to consider for computing the distribution.
                By Default (None), all available stimulus ids are considered.

        Returns:
            norm_dists (dict): {channel: (num_itr,)} True distribution of inter-trial correlations
            null_dists (dict): {channel: (num_itr,)} Null distribution of inter-trial correlations
        """
        if stim_ids is None:
            stim_ids = list(repeated_spikes.keys())
        channel_ids = list(repeated_spikes[stim_ids[0]].keys())
        total_trial_repeats = repeated_spikes[stim_ids[0]][channel_ids[0]].shape[0]
        trial_ids = np.arange(total_trial_repeats)
        if num_trials is not None:
            assert num_trials <= total_trial_repeats and num_trials >= 2, \
                "num_trials must be between 1 and {}".format(total_trial_repeats)
            trial_ids = np.random.choice(trial_ids, size=num_trials, replace=True)  # bootsraping step

        norm_dists = {ch: np.zeros((num_itr,)) for ch in channel_ids}
        null_dists = {ch: np.zeros((num_itr,)) for ch in channel_ids}
        
        for itr in range(num_itr):
            seq_U = {ch: [] for ch in channel_ids}
            seq_V = {ch: [] for ch in channel_ids}
            for stim_id in np.random.permutation(stim_ids):
                tr1, tr2 = np.random.choice(trial_ids, size=2, replace=False)   # distinct pair required
                for ch in channel_ids:
                    seq_U[ch].append(repeated_spikes[stim_id][ch][tr1])
                    seq_V[ch].append(repeated_spikes[stim_id][ch][tr2])
            for ch in channel_ids:
                U = np.concatenate(seq_U[ch], axis=0).squeeze()
                V = np.concatenate(seq_V[ch], axis=0).squeeze()

                # Correlation for tr-tr distribution
                corr_ch = NormalizerCalculator.safe_corrcoef(U, V) 
                norm_dists[ch][itr] = corr_ch
                # Correlation for Null distribution
                V_shifted = np.roll(V, V.shape[0]//2, axis=0)
                null_ch = NormalizerCalculator.safe_corrcoef(U, V_shifted) 
                null_dists[ch][itr] = null_ch
        return norm_dists, null_dists

    @staticmethod
    def safe_corrcoef(x, y):
        """ Computes the Pearson correlation coefficient between two arrays,
        handling cases where all elements are the same."""
        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return 0.0
        return np.corrcoef(x, y)[0, 1]


    ######################################################################
    ################## methods simplified: 06-24-25: Ends here...
    ######################################################################


    ######################################################################
    ################## methods redundant: 06-24-25: starts here...
    ######################################################################


        # ----------  Method 2 Null distribution: Randomly shifted seq.    ---------- #

    # def _compute_normalizer_null_dist_using_random_shifts(
    #         self, session, bin_width=50, num_itr=1000000, mVocs=False
    #         ):
    #     """Compute dist. of normalizer for correlations (repeatability of neural
    #     spikes), using 100k runs with random concatenation of trials at each iteration.."""
    #     stim_wise_repeated_spikes = self.get_stim_wise_repeated_spikes(
    #         session=session, bin_width=bin_width, mVocs=mVocs
    #         )
    #     normalizer_all = self.inter_trial_corr_using_random_pairing(
    #         stim_wise_repeated_spikes, num_itr=num_itr, circular_shift=True
    #         )
    #     # remove the nan entries from the distribution..
    #     normalizer_all = np.delete(normalizer_all, np.where(np.isnan(normalizer_all))[0], axis=0)
    #     return normalizer_all
    

    # --------   Method 1: computes normalizer separately for each channel   ------- #
    # --------                  using random pairs of trials.                ------- #

    # def _compute_normalizer_using_random_pairs(
    #         self, session, bin_width=50, num_itr=1000000, mVocs=False):
    #     """Compute dist. of normalizer for correlations (repeatability of neural
    #     spikes), using 100k runs with random concatenation of trials at each iteration.."""
    #     stim_wise_repeated_spikes = self.get_stim_wise_repeated_spikes(session=session, bin_width=bin_width, mVocs=mVocs)
    #     normalizer_all = self.inter_trial_corr_using_random_pairing(
    #         stim_wise_repeated_spikes, num_itr=num_itr, circular_shift=False
    #         )
    #     # remove the nan entries from the distribution..
    #     normalizer_all = np.delete(normalizer_all, np.where(np.isnan(normalizer_all))[0], axis=0)
    #     return normalizer_all


    ######################################################################
    ################## methods redundant: 06-24-25: Ends here...
    ######################################################################


    


    ### -------------------------------------------------------------------------- ###
    ###       Computing the distribution of Normalizers (repeated trials)          ###
    ### -------------------------------------------------------------------------- ###

    

    ### -------------------------------------------------------------------------- ###
    ###       Computing the distribution of Normalizers (repeated trials)          ###
    ### -------------------------------------------------------------------------- ###

    


    # --------   Method 2: computes normalizer separately for each channel   -------- #
    # --------                  using all possible pairs                     -------- #

    # def _compute_normalizer_all_possible_pairs(self, session, bin_width, mVocs=False):
    #     """Computes correlations separately for each sentence and using 
    #     all possible trial pairs. Returns the distribution comprising of 
    #     data points for all channels.    
    #     Args:
    #         bin_width: int = bin width in ms
    #         mVocs: bool = If True, mVocs trials are considered.

    #     Returns:
    #         corr_dist: ndarray = shape = (all_possible_pairs*num_sents, ch) 
    #     """
    #     stim_wise_repeated_spikes = self.get_stim_wise_repeated_spikes(
    #         session=session, bin_width=bin_width, mVocs=mVocs
    #         )
    #     corr_dist_combined = []
    #     for stim_id, stim_spikes in stim_wise_repeated_spikes.items():
    #         corr_dist_sent = self.inter_trial_corr_all_possible_pairs(stim_spikes)
    #         corr_dist_combined.append(corr_dist_sent)
    #     corr_dist_combined = np.concatenate(corr_dist_combined, axis=0)
    #     corr_dist_combined = np.nan_to_num(corr_dist_combined)
    #     return corr_dist_combined
    
    # --------    computes normalizer separately for each channel   -------- #
    # --------             using all possible pairs 	            -------- #
    

    # def get_normalizer_for_session(
    #         self, session, bin_width=20, delay=0, force_redo=False, mVocs=False,
    #         random_pairs=True, num_itr=1000,
    #         ):
    #     """Retrieves the distribution of normalizer for all the channels 
    #     of the specified session, at specified bin_width and delay.
    #     """
    #     session = str(int(float(session)))    # to make sure session is in str format.
    #     bin_width = int(bin_width)
    #     dataset_name = self.dataset_name
    #     logger.info(f"Getting normalizer dist. for sess-{session}, bw-{bin_width}, mVocs={mVocs}")

    #     if random_pairs:
    #         method = 'random'
    #     else:
    #         method = 'app'
        
    #     if dataset_name=='ucsf' and mVocs:
    #         # handling the excluded sessions for mVocs...
    #         if session == '190726':
    #             return np.zeros((1000, 60))
    #         elif session == '200213':
    #             return np.zeros((1000, 64))
            
    #     if not force_redo:
    #         norm_dist_session = io.read_normalizer_distribution(
    #             bin_width=bin_width, delay=delay, session=session, 
    #             method=method, mVocs=mVocs, dataset_name=dataset_name
    #             )
    #     if force_redo or (norm_dist_session is None):
    #         if random_pairs:
    #             norm_dist_session = self._compute_normalizer_using_random_pairs(
    #                 session, bin_width=bin_width, mVocs=mVocs, num_itr=num_itr
    #                 )
    #         else:
    #             norm_dist_session = self._compute_normalizer_all_possible_pairs(
    #                 session, bin_width=bin_width, mVocs=mVocs
    #                 )
    #         # save normalizer dist to disk..s
    #         io.write_normalizer_distribution(
    #             session, bin_width, delay, norm_dist_session, method=method, mVocs=mVocs, 
    #             dataset_name=dataset_name
    #         )

    #     # Redundant check: We have already penalized correlations for being NAN (zero sequence)
    #     # check zero valid samples in the distribution
    #     if norm_dist_session.shape[0] == 0:
    #         norm_dist_session = np.zeros((1, norm_dist_session.shape[-1]))
    #     return norm_dist_session



    ### --------------------------------------------------------------------------###
    ###         Computing the Null distribution of Normalizers                    ###
    ### --------------------------------------------------------------------------###

    # --------  Method 1 Null distribution: Using random poisson sequences   ------ #

    def _compute_normalizer_null_dist_using_poisson(
            self, bin_width, spike_rate=50, p_value=5, num_itr=10000, mVocs=False
        ):
        """Normalizer based on assumption that spikes are generated by a poisson process, 
        and are uniformly distributed in time."""
        logger.info(f"Poisson Process: Null distribution for bin_width: {bin_width}, spike_rate: {spike_rate}...")
        test_duration = self.get_testing_stim_duration(mVocs)
        logger.info(f"Test duration: {test_duration:.2f} sec")
        total_spikes = int(spike_rate * test_duration)
        num_bins = BaseDataset.calculate_num_bins(test_duration, bin_width/1000)
        null_dist = []
        for i in range(num_itr):
            spike_times_1 = np.random.uniform(0, test_duration, int(total_spikes))
            counts_1,_ = np.histogram(spike_times_1, num_bins)

            spike_times_2 = np.random.uniform(0, test_duration, int(total_spikes))
            counts_2, _ = np.histogram(spike_times_2, num_bins)

            null_dist.append(np.corrcoef(counts_1, counts_2)[0,1])
       
        q = 100 - p_value
        return np.percentile(null_dist, q), np.array(null_dist)
    

    def get_normalizer_null_dist_using_poisson(
            self, bin_width, spike_rate=50, num_itr=10000, force_redo=False,
            mVocs=False
        ):
        """Retrieves null distribution for normalizer using poisson
        sequences. Reads off the disk, if already available, or recomputes
        otherwise..
        
        Args:
            bin_width: int = bin width in ms
            spike_rate: int = spikes per second (Hz)
            itr: int = number of iterations.
            mVocs: bool = If True, 
        """
        dataset_name = self.dataset_name
        if not force_redo:
            null_dist_poisson = io.read_normalizer_null_distribution_using_poisson(
                bin_width=bin_width, spike_rate=spike_rate, mVocs=mVocs, dataset_name=dataset_name
                )
        if force_redo or null_dist_poisson is None:
            threshold, null_dist_poisson = self._compute_normalizer_null_dist_using_poisson(
                bin_width=bin_width, spike_rate=spike_rate, num_itr=num_itr, mVocs=mVocs
            )
            io.write_normalizer_null_distribution_using_poisson(
                bin_width, spike_rate, null_dist_poisson, mVocs=mVocs,
                dataset_name=dataset_name
            )
        return null_dist_poisson
    
    
    

    # def get_normalizer_null_dist_using_random_shifts(
    #         self, session, bin_width=50, mVocs=False, num_itr=100000, force_redo=False
    #     ):
    #     """Retrieves null distribution for normalizer using randomly 
    #     shifted spike sequence of a trial vs non-shifted sequence.
    #     At each iteration, trial is randomly choosen out of all available 
    #     trials.
    #     Reads off the disk, if already available, or recomputes
    #     otherwise..
        
    #     Args:
    #         bin_width: int = bin width in ms
    #         spike_rate: int = spikes per second (Hz)
    #         itr: int = number of iterations.
    #     """
    #     session = str(int(float(session)))  # enforcing exact format of session
    #     dataset_name = self.dataset_name
    #     null_dist_sess = io.read_normalizer_null_distribution_random_shifts(
    #         session, bin_width=bin_width, dataset_name=dataset_name
    #         )
    #     if null_dist_sess is None or force_redo:
    #         null_dist_sess = self._compute_normalizer_null_dist_using_random_shifts(
    #             session=session, bin_width=bin_width, num_itr=num_itr, mVocs=mVocs
    #         )
    #         io.write_normalizer_null_distribution_using_random_shifts(
    #             session, bin_width=bin_width, null_dist_sess=null_dist_sess,
    #             dataset_name=dataset_name
    #         )
    #     return null_dist_sess

        
    













# 	def _compute_significant_sessions_and_channels_using_poisson_null(
# 			self, bin_width = 50, spike_rate = 50, p_threshold = 0.01, mVocs=False
# 		):
# 		"""Computes significant channels for all the sessions, using
# 		two-sample Welch's t-test (assumes unqual variances).
# 		It test the null hypothesis that two distributions, distribution of repeatability
# 		correlations (normalizer dist.) and distribution of correlation between
# 		random poisson sequences (null dist.) have equal means.

# 		Args:
# 			bin_width: int = bin width in ms.
# 			spike_rate: int = spikes/second for poisson sequences in Null dist.
# 			p_threshold: float = threshold used to decide if null hypothesis if False.

# 		Returns:
# 			dict: {session: np.array([ch, ch])}
# 		"""
# 		total_sig_channels = 0
# 		sessions = self.metadata.get_all_available_sessions()
# 		significant_sessions_and_channels = {}
# 		null_dist = self.get_normalizer_null_dist_using_poisson(
# 			bin_width=bin_width, spike_rate=spike_rate, mVocs=mVocs
# 			)
# 		null_dist = np.array(null_dist)
# 		for session in sessions:
# 			norm_dist = self.get_normalizer_for_session_random_pairs(
# 				session=session, bin_width=bin_width, mVocs=mVocs 
# 			)
# 			num_channels = self.metadata.get_num_channels(session)
# 			for ch in range(num_channels):
# 				# pvalue = scipy.stats.ttest_ind(
# 				#     norm_dist[:,ch], null_dist, equal_var=False, alternative='greater'
# 				#     ).pvalue
# 				pvalue = scipy.stats.mannwhitneyu(
# 					norm_dist[:,ch], null_dist,alternative='greater'
# 					).pvalue

# 				# significance condition..
# 				if pvalue < p_threshold:    
# 					total_sig_channels += 1
# 					if session in significant_sessions_and_channels.keys():
# 						significant_sessions_and_channels[session].append(ch)
# 					else:
# 						significant_sessions_and_channels[session] = [ch]
# 		print(f"Total significant neurons at {bin_width}ms bin width = {total_sig_channels}")
# 		return significant_sessions_and_channels

# 	def get_significant_sessions_and_channels_using_poisson_null(
# 			self, bin_width = 50, spike_rate=50, p_threshold=0.05, force_redo=False,
# 			mVocs=False
# 		):
# 		"""Retrieves significant channels for all the sessions, using
# 		two-sample Welch's t-test (assumes unqual variances).
# 		Reads off the disk and recomputes if not found.

# 		Args:
# 			bin_width: int = bin width in ms.
            
# 		Returns:
# 			dict: {session: np.array([ch, ch])}
# 		"""
# 		significant_sessions_and_channels = io.read_significant_sessions_and_channels(
# 			bin_width=bin_width, p_threshold=p_threshold, mVocs=mVocs
# 			)
        
# 		if significant_sessions_and_channels is None or force_redo:
# 			significant_sessions_and_channels = self._compute_significant_sessions_and_channels_using_poisson_null(
# 				bin_width=bin_width, spike_rate=spike_rate, p_threshold=p_threshold, mVocs=mVocs
# 			)

# 			io.write_significant_sessions_and_channels(
# 				bin_width=bin_width,
# 				p_threshold=p_threshold,
# 				significant_sessions_and_channels=significant_sessions_and_channels,
# 				mVocs=mVocs
# 			)
# 		return significant_sessions_and_channels



    

        
        


# 	def _compute_significant_sessions_and_channels_using_shifts_null(
# 			self, bin_width = 50, p_threshold = 0.05, min_shift_frac=0.2, max_shift_frac=0.8,
# 		):
# 		"""Computes significant channels for all the sessions, using
# 		two-sample Welch's t-test (assumes unqual variances).
# 		It test the null hypothesis that two distributions, distribution of repeatability
# 		correlations (normalizer dist.) and distribution of correlation between
# 		spikes sequence and its randomly shifted version (null dist.) have equal means.

# 		Args:
# 			bin_width: int = bin width in ms.
# 			spike_rate: int = spikes/second for poisson sequences in Null dist.
# 			p_threshold: float = threshold used to decide if null hypothesis if False.

# 		Returns:
# 			dict: {session: np.array([ch, ch])}
# 		"""
# 		total_sig_channels = 0
# 		sessions = self.metadata.get_all_available_sessions()
# 		significant_sessions_and_channels = {}
# 		# null_dist = self.get_normalizer_null_dist_using_poisson(
# 		#     bin_width=bin_width, spike_rate=spike_rate
# 		#     )
# 		# null_dist = np.array(null_dist)
# 		for session in sessions:
# 			null_dist = self.get_normalizer_null_dist_using_random_shifts(
# 				session=session, bin_width=bin_width,
# 				min_shift_frac=min_shift_frac, max_shift_frac=max_shift_frac
# 				)

# 			norm_dist = self.get_normalizer_for_session_app(
# 				session=session, bin_width=bin_width 
# 			)
# 			num_channels = self.metadata.get_num_channels(session)
# 			for ch in range(num_channels):
# 				pvalue = scipy.stats.ttest_ind(
# 					norm_dist[:,ch], null_dist[:,ch], equal_var=False, alternative='greater'
# 					).pvalue
# 				# significance condition..
# 				if pvalue < p_threshold:    
# 					total_sig_channels += 1
# 					if session in significant_sessions_and_channels.keys():
# 						significant_sessions_and_channels[session].append(ch)
# 					else:
# 						significant_sessions_and_channels[session] = [ch]
# 		print(f"Total significant neurons at {bin_width}ms bin width = {total_sig_channels}")
# 		return significant_sessions_and_channels
    


# 	def get_significant_sessions_and_channels_using_shifts_null(
# 			self, bin_width = 50, p_threshold=0.05, force_redo=False
# 		):
# 		"""Retrieves significant channels for all the sessions, using
# 		two-sample Welch's t-test (assumes unqual variances).
# 		Reads off the disk and recomputes if not found.

# 		Args:
# 			bin_width: int = bin width in ms.
            
# 		Returns:
# 			dict: {session: np.array([ch, ch])}
# 		"""
# 		significant_sessions_and_channels = io.read_significant_sessions_and_channels(
# 			bin_width=bin_width, p_threshold=p_threshold, use_poisson_null=False
# 			)
        
# 		if significant_sessions_and_channels is None or force_redo:
# 			significant_sessions_and_channels = self._compute_significant_sessions_and_channels_using_shifts_null(
# 				bin_width=bin_width, p_threshold=p_threshold,
# 				min_shift_frac=0.2, max_shift_frac=0.8, # default values..
# 			)

# 			io.write_significant_sessions_and_channels(
# 				bin_width=bin_width,
# 				p_threshold=p_threshold,
# 				significant_sessions_and_channels=significant_sessions_and_channels,
# 				use_poisson_null=False
# 			)
# 		return significant_sessions_and_channels











# 	def get_normalizer_for_session_app(self, session, bin_width=20, delay=0, force_redo=False):
# 		"""Retrieves the distribution of normalizer for all the channels 
# 		of the specified session, at specified bin_width and delay.
# 		"""
# 		session = session = str(int(float(session)))    # to make sure session is in str format.
# 		bin_width = int(bin_width)
# 		norm_dict_all_sessions = io.read_normalizer_distribution(
# 			bin_width=bin_width, delay=delay, method='app',
# 			)
# 		if (norm_dict_all_sessions is None) or (session not in norm_dict_all_sessions.keys()) or force_redo:
# 			norm_dist_session = self._compute_normalizer_all_possible_pairs(
# 				session, bin_width, delay=delay
# 				)
# 			# save normalizer dist to disk..s
# 			io.write_normalizer_distribution(
# 				session, bin_width, delay, norm_dist_session, method='app',
# 			)
# 			return  norm_dist_session
# 		else:
# 			return norm_dict_all_sessions[session]













# 		# if normalizer_filename is None:
# 		# 	# print(f"Using default normalizer file...")
# 		# 	normalizer_filename = "modified_bins_normalizer.csv"
# 		# 	# "corr_normalizer.csv" original with full length sequnces. 
        
# 		# # print(f"Creating normalizer object from: {normalizer_filename}")
# 		# # self.dataframe, self.filepath = Normalizer.load_data(normalizer_filename)
# 		# self.metadata = NeuralMetaData()
# 		# self.loaded_datasets = {}
# 		# self.test_sent_IDs = [12,13,32,43,56,163,212,218,287,308]

# 	def get_significant_sessions(self, bin_width=20, delay=0, threshold=0.068):
# 		"""Returns a list of session having at least one significant channels"""
# 		select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
# 		select_data = select_data[select_data['normalizer'] >= threshold]
# 		return select_data['session'].unique()
    
# 	def get_good_channels(self, session, bin_width=20, delay=0, threshold=0.068):
# 		"""Returns a list of sig. channels for the session, bw and delay"""
# 		select_data = self.get_normalizer_for_session(session, bin_width=bin_width, delay=delay)
# 		select_data = select_data[select_data['normalizer'] >= threshold]
# 		return select_data['channel'].unique()
        
# 	def _get_normalizers_for_bin_width_and_delay(self, bin_width=20, delay=0):
# 		"""Retrieves section of dataframe for specified bin_width and delay"""
# 		select_data =  self.dataframe[
# 			(self.dataframe['bin_width']==bin_width) &\
# 			(self.dataframe['delay']==delay)
# 		]
# 		return select_data

# 	def get_normalizer_for_session(self, session, bin_width=20, delay=0):
# 		"""Retrives normalizer for the arguments."""
# 		session = float(session)
# 		select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
# 		select_data = select_data[(select_data['session']==session)]
# 		if select_data.shape[0] == 0:
# 			print(f"Results NOT available for session-{session} at bw-{bin_width} & delay-{delay}, computing now...", end='')
# 			self.save_normalizer_for_session(session, bin_width=bin_width, delay=delay)
# 			print(f"Done.")
# 			# recursive call after computing and saving resutlts
# 			self.get_normalizer_for_session(session, bin_width, delay)
# 			# raise ValueError(f"Results NOT available for session-{session} at bw-{bin_width} & delay-{delay},\
# 			#     use 'save_normalizer_for_session(...)' or 'save_normalizer_for_all_sessions()'")
# 		else:
# 			return select_data

    
# 	def save_normalizer_for_all_sessions(self, bin_width=20, delay=0):
# 		sessions = self.metadata.get_all_available_sessions()
# 		select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
# 		sessions_done = select_data['session'].unique()

# 		sessions_remaining = sessions[np.isin(sessions, sessions_done, invert=True)]

# 		for session in sessions_remaining:
# 			print(f"Normalizer not available for {session}" +
# 				f" at bw-{bin_width}, delay-{delay}, computing now...")
# 			strt_time = time.time()
# 			self.save_normalizer_for_session(session, bin_width=bin_width, delay=delay)
# 			end_time = time.time()
# 			print(f"It took {end_time-strt_time}s for session-{session}.!")


# 	def save_normalizer_for_session(self, session, bin_width=20, delay=0):
# 		"""Computes and saves the normalizer result, for the given configuration."""
# 		norm_dist = self._compute_normalizer(session, bin_width=bin_width,
# 									   delay=delay)
# 		session = float(session)
# 		num_channels = norm_dist.size
# 		data = np.stack([
# 			session*np.ones(num_channels),
# 			np.arange(num_channels),
# 			bin_width*np.ones(num_channels),
# 			delay*np.ones(num_channels),
# 			norm_dist
# 		], axis=1
# 		)
# 		df = pd.DataFrame(
# 			data= data,
# 			columns=self.dataframe.columns,
# 						)
# 		# writing back...
# 		self.dataframe = Normalizer.write_data(df, self.filepath)

# 		# self.dataframe = pd.concat([self.dataframe, df], axis=0, ignore_index=True)
# 		# self.dataframe.to_csv(self.filepath, index=False)
    





# #########   computes normalizer for the session ##############    
   

# 	# def _compute_normalizer(self, session, bin_width=20, delay=0, n=100000):
# 	#     """Compute dist. of normalizer for correlations (repeatability of neural
# 	#     spikes), and return median."""
# 	#     # session = str(int(session))
# 	#     # dataset = NeuralData(session)
# 	#     dataset = self._get_dataset_obj(session)

# 	#     sent_wise_repeated_spikes = {}
# 	#     for sent in self.metadata.test_sent_IDs:
# 	#         sent_wise_repeated_spikes[sent] = dataset.get_repeated_trials(
# 	#             sents=[sent], bin_width=bin_width, delay=delay
# 	#             )
            
# 	#     normalizer_all = Normalizer.inter_trial_corr_using_random_pairing(
# 	#         sent_wise_repeated_spikes, n=n
# 	#         )
# 	#     normalizer_all_med = np.median(normalizer_all, axis=0)
# 	#     return normalizer_all_med, normalizer_all
    
 



# 	def _get_dataset_obj(self, session):
# 		"""Retrieves dataset object"""
# 		session = str(int(session))
# 		if session not in self.loaded_datasets.keys():
# 			self.loaded_datasets.clear()
# 			self.loaded_datasets[session] = NeuralData(session)
# 		return self.loaded_datasets[session]
    
# 	def compute_normalizer_threshold(self, bin_width, p_value=5, itr=10000):
# 		"""Computes significance threshold for normalizer at bin_width."""

# 		total_samples_test_set = 0
# 		if bin_width == 1000:
# 			# special case...
# 			total_samples_test_set = self.metadata.test_sent_IDs.size
# 		else:
# 			for sent in self.metadata.test_sent_IDs:
# 				total_samples_test_set += self.metadata.stim_samples(sent, bin_width=bin_width)

# 		print(f"Computing null distribution for bin_width: {bin_width}, num_samples: {total_samples_test_set}...")
# 		null_dist = []
# 		for i in range(itr):
# 			gaussian_sample_of_same_length = np.random.randn(2, total_samples_test_set)
# 			null_dist.append(np.corrcoef(gaussian_sample_of_same_length)[0,1])

# 		q = 100 - p_value
# 		return np.percentile(null_dist, q), null_dist
    

    




    

        

    
#########################################################
######      static methods
########################################################

    # @staticmethod
    # def write_data(df, filepath):
    #     if os.path.exists(filepath):
    #         existing_data = pd.read_csv(filepath)
    #     else:
    #         existing_data = pd.DataFrame(columns=df.columns)
    #     updated_data = pd.concat([existing_data, df], axis=0, ignore_index=True)
    #     updated_data.to_csv(filepath, index=False)
    #     print(f"Result updated...!")
    #     return updated_data
        


    # @staticmethod
    # def load_data(filename):
    #     """Loads/creates dataframe containing normalizers"""
    #     filepath = os.path.join(
    #         results_dir, 'neural_repeatibitliy', filename
    #         )
    #     if not os.path.exists(os.path.dirname(filepath)):
    #         print(f"Creating directory structure for normalizer...!")
    #         os.makedirs(os.path.dirname(filepath))

    #     if os.path.exists(filepath):
    #         print(f"Reading existing dataframe.")
    #         dataframe = pd.read_csv(filepath)
    #     else:
    #         print(f"Creating new dataframe.")
    #         columns = ['session', 'channel','bin_width',
    #                    'delay', 'normalizer']
    #         dataframe = pd.DataFrame(columns=columns)
    #     return dataframe, filepath
    
    # @staticmethod
    # def inter_trial_corr_all_possible_pairs(spikes_all_trials, num_trials=None):
    #     """Computes correlations between all possible trial pairs,
    #     and returns the distribution. For example for input of shape (11,:, 64),
    #     there are 11 trials, so number of possible pairs are 11C2 = 11*10/2 = 55.
        
    #     Args:
    #         spikes_all_trials: ndarray = (num_trials, seq_length, channels)

    #     Returns:
    #         sent_corr_dist: ndarray = shape = (all_possible_pairs, ch) 
    #     """
    #     if num_trials is None:
    #         num_trials = spikes_all_trials.shape[0]
    #     sent_corr_dist = []
    #     num_channels = spikes_all_trials.shape[-1]
    #     # imagine upper half of the matrix, with diagonal entries included..
    #     possible_pairs = int((num_trials**2 - num_trials)/2)
    #     sent_corr_dist = np.zeros((possible_pairs, num_channels))
    #     index = 0
    #     for i in range(num_trials):
    #         for j in range(i+1, num_trials):
    #             for ch in range(num_channels):
    #                 trial1 = spikes_all_trials[i,:,ch]
    #                 trial2 = spikes_all_trials[j,:,ch]
    #                 sent_corr_dist[index, ch] = np.corrcoef(
    #                     trial1, trial2
    #                     )[0,1]
    #             index += 1
    #     return sent_corr_dist

    # @staticmethod
    # def inter_trial_corr(spikes, n=100000):
    #     """Compute distribution of inter-trials correlations.

    #     Args: 
    #         spikes (ndarray): (repeats, samples/time, channels)

    #     Returns:
    #         trials_corr (ndarray): (n, channels) distribution of inter-trial correlations
    #     """
    #     num_channels = spikes.shape[-1]
    #     trials_corr = np.zeros((n, num_channels))
    #     for t in range(n):
    #         trials = np.random.choice(np.arange(0,spikes.shape[0]), size=2, replace=False)
    #         for ch in range(num_channels):
    #             trials_corr[t, ch] = np.corrcoef(
    #                 spikes[trials[0],:,ch].squeeze(), spikes[trials[1],:,ch].squeeze()
    #                 )[0,1]
    #     return trials_corr

    









        # num_channels = next(iter(sent_wise_repeated_spikes.values())).shape[-1]
        # trials_corr = np.zeros((num_itr, num_channels))
        # stim_ids = list(sent_wise_repeated_spikes.keys())
        # num_trials, seq_lens = sent_wise_repeated_spikes[stim_ids[0]].shape[:2]
        # trial_ids = np.arange(num_trials)
        # for t in range(num_itr):
        #     long_seq1 = []
        #     long_seq2 = []
        #     for stim_id in np.random.permutation(stim_ids):
        #         spikes = sent_wise_repeated_spikes[stim_id]
        #         tr1, tr2 = np.random.choice(trial_ids, size=2, replace=False)

        #         long_seq1.append(spikes[tr1])
        #         long_seq2.append(spikes[tr2])
        #     long_seq1 = np.concatenate(long_seq1, axis=0)
        #     long_seq2 = np.concatenate(long_seq2, axis=0)

        #     if circular_shift:
        #         # randomly roll the sequence to get the second sequence
        #         long_seq2 = np.roll(long_seq2, seq_lens//2, axis=0)

        #     for ch in range(num_channels):
        #         corr_ch = NormalizerCalculator.safe_corrcoef(
        #             long_seq1[...,ch].squeeze(), long_seq2[...,ch].squeeze()
        #         )   

        #         # corr_ch = np.corrcoef(
        #         #     long_seq1[...,ch].squeeze(), long_seq2[...,ch].squeeze()
        #         #     )[0,1]
        #         # nan might result from entires sequence being zero, 
        #         # penalize that by setting corr equal to zero.
        #         # if np.isnan(corr_ch):
        #         #     corr_ch = 0
        #         trials_corr[t, ch] = corr_ch
        # return trials_corr
    
     


    


