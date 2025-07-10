
import numpy as np

from .base_dataset import BaseDataset, create_neural_dataset
from .base_metadata import create_neural_metadata
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
            # trial_ids = np.random.choice(trial_ids, size=num_trials, replace=True)  
            trial_ids = NormalizerCalculator.sample_subset_of_trials(trial_ids, num_trials) # bootsraping step

        norm_dists = {ch: np.zeros((num_itr,)) for ch in channel_ids}
        null_dists = {ch: np.zeros((num_itr,)) for ch in channel_ids}
        
        for itr in range(num_itr):
            seq_U = {ch: [] for ch in channel_ids}
            seq_V = {ch: [] for ch in channel_ids}
            for stim_id in np.random.permutation(stim_ids):
                tr1, tr2 = np.random.choice(trial_ids, size=2, replace=False)   # distinct pair required
                while tr1 == tr2:   # making sure trials are distinct, even when input trial_ids have repeated trials
                    tr1, tr2 = np.random.choice(trial_ids, size=2, replace=False)
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
    
    @staticmethod
    def sample_subset_of_trials(trial_ids, num_trials):
        """ Samples a subset of trials from the given trial_ids. 
        Makes sure that the sampled subset has at least 2 unique trials.
        Args:
            trial_ids: array-like = Array of trial ids to sample from.
            num_trials: int = Number of trials to sample.
        """
        subset = np.random.choice(trial_ids, size=num_trials, replace=True)  # bootsraping step
        while len(np.unique(subset)) < 2:
            # If the sampled subset has less than 2 unique trials, resample
            subset = np.random.choice(trial_ids, size=num_trials, replace=True)
        return subset
    
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
    