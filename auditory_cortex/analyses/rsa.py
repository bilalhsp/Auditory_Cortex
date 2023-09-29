

import numpy as np


from auditory_cortex.models import Regression
from auditory_cortex.analysis import Correlations


class RSA:

    def __init__(self, model_name) -> None:
        self.model = Regression(model_name)
        self.corr_obj = Correlations(model_name=model_name+'_'+'opt_neural_delay')

    @staticmethod
    def equalize_seq_lengths(rep1, rep2, **kwargs):
        """Makes sequence lengths equal, also makes sure num_features of both 
        sequences are equal.
        """
        if 'clip' in kwargs:
            clip = kwargs.pop('clip')
        else:
            clip = False

        if len(kwargs) != 0: 
            raise ValueError("You have provided unrecognizable keyword args")
        
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
              
        Returns:
            float [0,1] = value of pearon's correlation.
        """
        rep1, rep2 = RSA.equalize_seq_lengths(rep1, rep2, **kwargs)
        return np.corrcoef(rep1.reshape(-1), rep2.reshape(-1))[0,1]

        # # pearson's correlation...
        # x = rep1 - np.mean(rep1)
        # y = rep2 - np.mean(rep2)
        # corr = np.mean(x*y)
        # pearson_corr = corr/(np.std(rep1)*np.std(rep2))
        
        # # confine pearson's correlation in range [0,1]
        # pearson_corr = max(0, pearson_corr)
        # pearson_corr = min(1, pearson_corr)
        # return pearson_corr
    
    @staticmethod
    def compute_similarity_matrix(Z_stim, **kwargs):
        """Computes RSA matrix, given the stimulus representations (Z_stim) 
        as argument.
        
        Args:
            Z_stim: dict = dictionary of representations indexed by stimulus ID's.
            
        Returns:
            ndarray = (num_stimulus x num_stimulus) matrix of pairwise similarities.
        """
        N = len(Z_stim)
        rsa_matrix = np.zeros((N,N))
        for i, rep1 in enumerate(Z_stim.values()):
            for j, rep2 in enumerate(Z_stim.values()):
                rsa_matrix[i,j] = RSA.compute_similarity(rep1, rep2, **kwargs)

        return rsa_matrix 

    def get_rsa_matrix(self, layer=6, **kwargs):
        """Computes and returns similarity matrix for specified layer of 
        the model. 

        Args:
            layer: int = ID of the layer to get the matrix for.
            dissimilarity: bool = if True, computes dissimilarity (1-r) instead
                of similiarity (r).
        
        """

        Z_stim = self.model.sampled_features[layer]
        matrix = RSA.compute_similarity_matrix(Z_stim, **kwargs)
        return matrix
    
    def get_neural_rsa(self, session, **kwargs):


        session = str(int(session))
        self.model._load_dataset_session(session)
        self.model.get_dataset_object(session).extract_spikes(20, 0)#, sents=sents)
        spikes = self.model.get_dataset_object(session).raw_spikes

        # keep only good channels...
        good_channels = self.corr_obj.get_good_channels(session, threshold=0.068)
        good_channels = np.array(good_channels, dtype=np.uint32)
        Z_stim = {sent: array[...,good_channels] for sent, array in spikes.items()}

        matrix = RSA.compute_similarity_matrix(Z_stim, **kwargs)

        return matrix
            
        