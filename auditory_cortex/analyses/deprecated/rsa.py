import scipy
import numpy as np
import matplotlib.pyplot as plt

# from auditory_cortex.models import Regression
from auditory_cortex.deprecated.dataloader import DataLoader
from auditory_cortex.analyses.deprecated.regression_correlations import Correlations
from auditory_cortex.neural_data.deprecated.neural_meta_data import NeuralMetaData
from auditory_cortex.io_utils.io import write_RDM, read_RDM
from auditory_cortex.io_utils.io import write_cached_RDM_correlations
from auditory_cortex.io_utils.io import read_cached_RDM_correlations

# from auditory_cortex.io_utils.io import read_cached_spikes, write_cached_spikes
# from auditory_cortex.io_utils.io import read_cached_features, write_cached_features


class RSA:

    def __init__(self, model_name, identifier='global') -> None:
        self.model_name = model_name
        self.identifier = identifier
        # self.corr_obj = Correlations(model_name=model_name+'_'+'opt_neural_delay')
        # needed only for sessions and channels selection...!
        self.corr_obj = Correlations('wav2letter_modified_normalizer2')
        self.dataloader = DataLoader()
        self.metadata = NeuralMetaData()
        # self.model = None
        # self.raw_features = None
        # self.layer_ids = None
        # if load_features:
        #     self.create_regression_obj()

        # self.features = self.model.sampled_features



    @staticmethod
    def equalize_seq_lengths(rep1, rep2, **kwargs):
        """Makes sequence lengths equal, also makes sure num_features of both 
        sequences are equal.

        kwargs:
            clip: bool = Specifies how to equalize seq lengths, i.e. clip or pad.
                Default=False.

        """
        if 'clip' in kwargs:
            clip = kwargs.pop('clip')
        else:
            clip = False

        # if len(kwargs) != 0: 
        #     raise ValueError("You have provided unrecognizable keyword args")
        
        if rep1.ndim == 1:
            rep1 = np.expand_dims(rep1, axis=1)
        if rep2.ndim == 1:
            rep2 = np.expand_dims(rep2, axis=1)

        (seq_len1, num_feats_1)  = rep1.shape
        (seq_len2, num_feats_2)  = rep2.shape

        assert num_feats_1 == num_feats_2, "num_features of representations being compared must match."

        # Zero-pad or clip ...
        if seq_len1 > seq_len2:
            if clip:
                rep1 = rep1[:seq_len2]
            else:
                diff_len = seq_len1 - seq_len2
                zero_pad_seq = np.zeros((diff_len, num_feats_2))
                rep2 = np.concatenate([rep2, zero_pad_seq], axis=0)

        elif seq_len1 < seq_len2:
            if clip:
                rep2 = rep2[:seq_len1]
            else:
                diff_len = seq_len2 - seq_len1
                zero_pad_seq = np.zeros((diff_len, num_feats_1))
                rep1 = np.concatenate([rep1, zero_pad_seq], axis=0)
        return rep1, rep2

    @staticmethod
    def compute_similarity(rep1, rep2, **kwargs):
        """Computes similarity between input representations,
        rep1 and rep2. Input representations can be sequences
        different lengths (because audio stimuli could be of different
        lengths). Different lengths are handled by zero-padding the 
        shorter sequence.

        Args:
            rep1, rep2: ndarray = stimulus representation of the form (time, num_features)
                if rep1.ndim == 1 --> num_features = 1 

        kwargs:
            clip: bool = Specifies how to equalize seq lengths, i.e. clip or pad.
                Default=False.
              
        Returns:
            float [0,1] = value of pearon's correlation.
        """
        rep1, rep2 = RSA.equalize_seq_lengths(rep1, rep2, **kwargs)
        return np.corrcoef(rep1.flatten(), rep2.flatten())[0,1]

    @staticmethod
    def compute_similarity_matrix(Z_stim, identifier='', **kwargs):
        """Computes RSA matrix, given the stimulus representations (Z_stim) 
        as argument.
        
        Args:
            Z_stim: dict = dictionary of representations indexed by stimulus ID's.
            identifier: str = specifies how to equalize number of samples (time axis),
                of individual sequences. Choose from ['', 'global', 'average'].
                'global' clips all sequences at the length of the shortest seq.
                'average' collapses time axis by averaging (or RMS).
                '' clips all sequences pairwise during similarity compututation.
        kwargs:
            clip: bool = Specifies how to equalize seq lengths, i.e. clip or pad.
                Default=False.
            
        Returns:
            ndarray = (num_stimulus x num_stimulus) matrix of pairwise similarities.
        """
        if identifier == '':
            N = len(Z_stim)
            rsa_matrix = np.zeros((N,N))
            for i, rep1 in enumerate(Z_stim.values()):
                for j, rep2 in enumerate(Z_stim.values()):
                    rsa_matrix[i,j] = RSA.compute_similarity(rep1, rep2, **kwargs)
            return rsa_matrix 
    
        elif 'global' in identifier:
            mat_X = RSA.get_matrix_with_global_min_seq_len(Z_stim)
        elif 'average' in identifier:
            mat_X = RSA.get_matrix_with_seq_RMS(Z_stim)
            # mat_X = np.stack([np.mean(val, axis=0) for val in Z_stim.values()], axis=0)
        else:
            raise ValueError("Please specify 'identifier' correctly.")
        rsa_matrix = RSA.compute_RSM(mat_X)
        return rsa_matrix 
    
############################################################
##########  Moved to dataloader...

    # def create_regression_obj(self):
    #     self.model = Regression(self.model_name, load_features=False)
    #     self.layer_ids = self.model.layer_ids
    #     self.feature_dict = {}

    # def load_raw_features(self):
    #     """Loads raw features for the 'model', starts by attempting to
    #     read cached features, if not found, extract features and also
    #     cache them, for future use.
    #     """
    #     self.raw_features = read_cached_features(self.model_name)
    #     if self.raw_features is None:
    #         self.model.load_features(resample=False)
    #         self.raw_features = self.model.sampled_features
    #         # cache features for future use...
    #         write_cached_features(self.model_name, self.raw_features)
    
    # def get_layer_features(self, layer, bin_width=20):
    #     """Loads layer features, sampled according to bin_width,
    #     stores the results in a dict (bin_width) as key, so that 
    #     the features can be reused for rest of the layers (at the same bin_width).
        
    #     Args:
    #         layer: int = ID of the layer to get the features for.
    #         bin_width: int = bin width in ms.

    #     Returns:
    #         features for the layer
    #     """
    #     if self.model is None:
    #         # creates model and saves raw featuers..
    #         self.create_regression_obj()
    #     if self.raw_features is None:
    #         self.load_raw_features() 

    #     layer_ind = self.model.get_layer_index(layer)
    #     if bin_width not in self.feature_dict.keys():
    #         # print(f"Resampling raw features for bin_width={bin_width}..")
    #         features = self.model.resample(
    #             self.raw_features, bin_width=bin_width
    #             )
    #         self.feature_dict[bin_width] = features
        
    #     return self.feature_dict[bin_width][layer_ind]

    def compute_layer_RDM(self, layer=6, bin_width=20, **kwargs):
        """Computes and returns similarity matrix for specified layer of 
        the model. 

        Args:
            layer: int = ID of the layer to get the matrix for.
            dissimilarity: bool = if True, computes dissimilarity (1-r) instead
                of similiarity (r).
        kwargs:
            clip: bool = Specifies how to equalize seq lengths, i.e. clip or pad.
                Default=False.
        """
        layer = int(layer)
        # z_stim = self.get_layer_features(layer, bin_width=bin_width)
        z_stim = self.dataloader.get_DNN_layer_features(
            self.model_name, layer_ID=layer, bin_width=bin_width)
        print(f"Computing RDM for layer-{layer} of {self.model_name} ")
        matrix = RSA.compute_similarity_matrix(
            z_stim, identifier=self.identifier, **kwargs
            )
        matrix = np.clip(matrix, 0, 1)
        return 1-matrix
    
    #########################################
    #####  Moved to dataloader..


    # def get_neural_spikes(self, bin_width=20, threshold=0.068, **kwargs):
    #     """"Retrieves caches neural spikes, extract spikes if not already
    #     cached.

    #     Args:
    #         bin_width: int= bin_width in ms

    #     kwargs:
    #         area: str = ['core', 'belt'], default=None.
    #     """
    #     if 'area' in kwargs:
    #         area = kwargs.pop('area')
    #         if area != 'all':
    #             assert area in ['core', 'belt'], "Incorrect brain" + \
    #                 "area specified, choose from ['core', 'belt']"
    #     else:
    #         area = 'all'

    #     spikes = read_cached_spikes(bin_width=bin_width, threshold=threshold)
    #     if spikes is None or area not in spikes.keys():
    #         z_stim = self.get_sig_neural_data(
    #             threshold=threshold, bin_width=bin_width, area=area
    #             )
    #         write_cached_spikes(z_stim, bin_width=bin_width, area=area, threshold=threshold)
    #     else:
    #         z_stim = spikes[area]
    #     return z_stim 
    
    # def get_sig_neural_data(self, threshold=0.068, bin_width=20, **kwargs):
    #     """Get stimulus-wise neural spike data, for group of session 
    #     (core, belt or all), with significant sessions from these
    #     sessions concatenated together for each stimulus.

    #     Args:
    #         threshold: float = significance threshold (0,1).
    #         bin_width: int= bin_width in ms

    #     kwargs:
    #         area: str = ['core', 'belt'], default=None.
        
    #     Returns:
    #         dict: dictionary of stimulus representations, with sent
    #             ID's as keys.
    #     """
    #     if 'area' in kwargs:
    #         area = kwargs.pop('area')
    #         if area != 'all':
    #             assert area in ['core', 'belt'], "Incorrect brain" + \
    #                 "area specified, choose from ['core', 'belt']"
    #     else:
    #         area = 'all'
    #     # if len(kwargs) != 0: raise ValueError("Unrecognizable keyword args.")
    #     sig_sessions = self.corr_obj.get_significant_sessions(threshold=threshold)
    #     if area != 'all':
    #         sessions = self.metadata.get_all_sessions(area)
    #         sig_sessions = sig_sessions[np.isin(sig_sessions, sessions)]
    #     z_stim = {}
    #     if self.model is None:
    #         self.create_regression_obj()

    #     for session in sig_sessions:
    #         spikes_sess = self.model.get_raw_neural_spikes(session, bin_width=bin_width)

    #         # keep only good channels...
    #         good_channels = np.array(
    #             self.corr_obj.get_good_channels(session, threshold=threshold),
    #             dtype=np.uint32
    #             )
    #         if len(z_stim) != 0:
    #             z_stim = {
    #                 sent: np.concatenate([z_val, z_sess_val[...,good_channels]], axis=1) \
    #                 for (sent, z_val), z_sess_val in zip(z_stim.items(), spikes_sess.values())}
    #         else:
    #             z_stim = {sent: array[...,good_channels] for sent, array in spikes_sess.items()}

    #     return z_stim

    
    def compute_neural_RDM(self, bin_width=20, threshold=0.061, **kwargs):
        """Computes representational similarity matrix for neural data from 
        area (kwarg) specified.

        kwargs:
            area: str = ['core', 'belt'], default=None.
            clip: bool = Specifies how to equalize seq lengths, i.e. clip or pad.
                Default=False.

        Returns:
            ndarray = num_stim x num_stim matrix  
        """
        
        print(f"Computing Neural RSA matrix...")
        # z_stim = self.get_sig_neural_data(bin_width=bin_width, **kwargs)
        # z_stim = self.get_neural_spikes(bin_width=bin_width, **kwargs)
        z_stim = self.dataloader.get_all_neural_spikes(
            bin_width=bin_width, threshold=threshold,
            **kwargs
            )
        matrix = RSA.compute_similarity_matrix(
            z_stim, identifier=self.identifier, **kwargs
            )
        matrix = np.clip(matrix, 0, 1)
        return 1-matrix
            

    def get_layer_corr(
            self, layer, area='all', bin_width=20,
            iterations = 100, size=499,
            spearman_rank=True,
            redo_RDMs=False   
        ):
        # layer = self.model.get_layer_index(layer)
        layer_rdm = self.get_RDM(layer,
                bin_width=bin_width, force_redo=redo_RDMs)
        neural_rdm = self.get_RDM(area, neural=True,
                bin_width=bin_width, force_redo=redo_RDMs)
        
        corr_dist = RSA.get_RDM_similarity_dist(layer_rdm, neural_rdm,
                iterations=iterations, size=size, spearman_rank=spearman_rank)
        return corr_dist
        # corr_dist = []
        # for i in range(iterations):
        #     indices = np.random.choice(
        #         np.arange(layer_rdm.shape[0]), size, replace=True
        #         )
        #     # slicing 2d sub-matrix from the original matrix
        #     layer_sub_rdm = layer_rdm[indices][:,indices]
        #     neural_sub_rdm = neural_rdm[indices][:,indices]
        #     corr_dist.append(RSA.get_RDM_similarity(
        #         layer_sub_rdm, neural_sub_rdm, spearman_rank
        #         ))
            
        # return np.array(corr_dist)

    
    def get_layer_wise_corr(
            self, area='all', bin_width=20,
            iterations=100, size=499, spearman_rank=True,
            force_redo=False, verbose=False
            ):
        """Computes distribution of correlations for each layer of the
        network.

        Args:
            area: str = neural area selection, choices=['all','core','belt']
            bin_width: int = bin_width in ms.
            iterations: int = number of repeats to compute corr
            size: int = size of each sub-samples
            spearman_rank: bool = Default=True, Report spearman rank corr,
                pearson correlation if False.
        Returns:
            dict of array: each value in dict is distribution for a layer.
        """
        # corr_dict = {}
        # if self.layer_ids is None:
        #     self.create_regression_obj()
        # for layer_id in self.layer_ids:
            
        #     corr_dict[layer_id] = self.get_layer_corr(
        #         layer_id, area=area, bin_width=bin_width,
        #         iterations=iterations, size=size, spearman_rank=spearman_rank
        #     )
        
        # return corr_dict
        corr_dict = read_cached_RDM_correlations(self.model_name, self.identifier, area, bin_width)
        if corr_dict is None or force_redo:
            neural_rdm = self.get_RDM(area, neural=True, bin_width=bin_width,
                                            verbose=verbose)
            # neural_arr = RSA.get_lower_traingle(neural_mat)
            
            rsa_dict = read_RDM(self.model_name, identifier=self.identifier,
                                bin_width=bin_width, verbose=verbose)
            if rsa_dict is None:
                print(f"Computing RDM for network layers...!")
                rsa_dict = {}
                for layer_ID in self.dataloader.get_layer_IDs(self.model_name):
                    rsa_dict[layer_ID] = self.get_RDM(layer_ID, bin_width=bin_width, verbose=verbose)
                
                # raise FileNotFoundError("Make sure RDMs are saved for the NN configuration...!")
            rsa_dict = dict(sorted(rsa_dict.items(), key=lambda item: int(item[0])))
            corr_dict = {}
            for layer_id, layer_rdm in rsa_dict.items():
                corr_dict[layer_id] = RSA.get_RDM_similarity_dist(layer_rdm, neural_rdm,
                                            iterations=iterations, size=size,
                                            spearman_rank=spearman_rank)


            write_cached_RDM_correlations(corr_dict, self.model_name, self.identifier, area, bin_width)
        return corr_dict
    
    def get_RDM(
            self, key, neural=False, force_redo=False,
            bin_width=20, verbose=True
        ):
        """Retrieves, computes or force-recomputes RSA matrix. If RSA matrix 
        does not already exist at the disk location, or 'force_redo=True, computes
        RSA matrix for the given setting. Otherwise simply retrieves the matrix.

        Args:
            key: int or str= layer ID in case of model or ['core', 'belt', 'all]
                in case of neural.
            neural: bool = specifies if neural RSA needs to be done.
            force_redo: bool = forces to recompute RSA, even if results already
                saved to disk.
        Returns:
            ndarray = RSA matrix                     
        """
        if neural:
            model_name = 'neural'
        else:
            model_name = self.model_name
        # key = str(key)
        rsa_dict = read_RDM(model_name, identifier=self.identifier,
                                     bin_width=bin_width, verbose=verbose)
        if rsa_dict is None or key not in rsa_dict.keys() or force_redo:
            if neural:
                print(f"For area-{key}")
                matrix = self.compute_neural_RDM(
                    bin_width=bin_width, threshold=0.061,
                    area=key, clip=True
                )
            else:
                key = int(key)
                matrix = self.compute_layer_RDM(layer=key, bin_width=bin_width, clip=True)
            write_RDM(model_name, key, matrix, identifier=self.identifier,
                               bin_width=bin_width)
        else:
            matrix = rsa_dict[key]
        return matrix
    
    def visualize_RDM(self, key, neural=False, bin_width=20,
                ranked=True, cmap='turbo'):

        rdm = self.get_RDM(key=key, neural=neural, bin_width=bin_width)
        if ranked:
            ranked = rdm.argsort().argsort()
            ranked = ranked/ranked.max()
        else:
            ranked = rdm
        map = plt.imshow(ranked, cmap=cmap, interpolation='none')
        plt.colorbar(map)
        if neural:
            model = 'neural'
        else:
            model = self.model_name
        plt.title(f"{model}-{key}, {bin_width}ms")
            



###################################################################
    @staticmethod
    def compute_RSM(matrix_X):
        """Computes Representational Similarity matrix (RSM).
        
        Args:
            matrix_X: ndarray = (num_stim, num_samples) 
                where num_samples = time x channels.
        
        Returns:
            (num_stim x num_stim) matrix of correlations..
        """
        means = np.mean(matrix_X, axis=1)
        matrix_X = matrix_X - means[:,None]
        cov_X = np.matmul(matrix_X, matrix_X.T)

        # outer product of std. deviations...(to get matrix of normalizers)
        std_deviations = np.sqrt(np.diag(cov_X))
        normalizers = std_deviations[:,None] @ std_deviations[None, :]

        return cov_X/normalizers
    
    @staticmethod
    def get_matrix_with_global_min_seq_len(z_stim):
        # X is dict of 2d arrays (t, num_features)
        min_samples = 1000
        demean_z_stim = {}
        for sent_ID, val in z_stim.items():
            if val.shape[0]<min_samples:
                min_samples = val.shape[0]
            # demean sequence for every channel...
            demean_z_stim[sent_ID] = val -  np.mean(val, axis=0)[None,:]
            # val = val -  np.mean(val, axis=0)[None,:]

        # print(min_samples)
        # matrix_X = np.stack([val[:min_samples].flatten() for val in z_stim.values()], axis=0)
        matrix_X = np.stack([val[:min_samples].flatten() for val in demean_z_stim.values()], axis=0)
        return matrix_X
    
    @staticmethod
    def get_matrix_with_seq_RMS(z_stim):
        """Collapses time axis by RMS for each feature,

        Args:
            z_stim: dict = dictionary of representations indexed by stimulus ID's.
                where shape of each rep. is (time, features).

        Returns:
            matrix of dim (num_stim, num_features)
        """
        matrix_x = []
        for val in z_stim.values():
            matrix_x.append(np.sqrt(np.mean(val**2, axis=0)))
        return np.array(matrix_x) 
    
    @staticmethod
    def get_lower_traingle(mat):
        return mat[np.tril_indices(mat.shape[0], -1)]
    
    @staticmethod
    def get_RDM_similarity(layer_rdm, neural_rdm, spearman_rank=True):
        """Computes similarity of two RDMs provided as arguments.
        
        Args:
            layer_rdm: ndarray: symmetric, RDM matrix for layer
            neural_rdm: ndarray: symmetric, RDM matrix for neural region
        
        Returns:
            spearman_rank correlation or pearson correlation.
        """
        layer_arr = RSA.get_lower_traingle(layer_rdm)
        neural_arr = RSA.get_lower_traingle(neural_rdm)
        if spearman_rank:
            return scipy.stats.spearmanr(layer_arr, neural_arr)[0]
        else:
            return np.corrcoef(layer_arr, neural_arr)[0,1]

    @staticmethod
    def get_RDM_similarity_dist(
        layer_rdm, neural_rdm, iterations = 20, size=350,
            spearman_rank=True
        ):
        """Computes distribution of similarity of
        two RDMs provided as arguments.
        
        Args:
            layer_rdm: ndarray: symmetric, RDM matrix for layer
            neural_rdm: ndarray: symmetric, RDM matrix for neural region
            iterations: int = number of repeats to compute corr
            size: int = size of each sub-samples
            spearman_rank: bool = Default=True, Report spearman rank corr,
                pearson correlation if False.
            
        Returns:
            distribution of spearman_rank correlation or pearson correlation.
        """
        corr_dist = []
        for _ in range(iterations):
            indices = np.random.choice(
                np.arange(layer_rdm.shape[0]), size, replace=True
                )
            # slicing 2d sub-matrix from the original matrix
            layer_sub_rdm = layer_rdm[indices][:,indices]
            neural_sub_rdm = neural_rdm[indices][:,indices]
            corr_dist.append(RSA.get_RDM_similarity(
                layer_sub_rdm, neural_sub_rdm, spearman_rank
                ))   
        return np.array(corr_dist)
    
    @staticmethod
    def get_RDM_similarity_null_dist(
        layer_rdm, neural_rdm, iterations = 1000,
            spearman_rank=True
        ):
        """Computes NULL distribution of similarity of
        two RDMs provided as arguments, for statistical significance.
        
        Args:
            layer_rdm: ndarray: symmetric, RDM matrix for layer
            neural_rdm: ndarray: symmetric, RDM matrix for neural region
            iterations: int = number of repeats to compute corr
            spearman_rank: bool = Default=True, Report spearman rank corr,
                pearson correlation if False.
            
        Returns:
            distribution of spearman_rank correlation or pearson correlation.
        """
        corr_dist = []
        indices = np.arange(layer_rdm.shape[0])
        for _ in range(iterations):
            # shuffle in-place..
            np.random.shuffle(indices)
            # slicing 2d shuffled matrix from the original matrix
            layer_rdm_shuffled = layer_rdm[indices][:,indices]
            corr_dist.append(RSA.get_RDM_similarity(
                layer_rdm_shuffled, neural_rdm, spearman_rank
                ))   
        return np.array(corr_dist)
