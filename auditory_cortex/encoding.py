"""
TRF Module for Neural Encoding with Time-Delayed Regression Models

This module contains the implementation of a TRF (Temporal Response Function) class for performing 
neural encoding using time-delayed linear regression models. It includes functionality for 
cross-validated model fitting, evaluation of model performance, and optimization of hyperparameters 
such as lag and regularization strength. The module also includes a GPU-accelerated version of 
the TRF model, built on top of naplib's encoding functionality.

Classes:
    TRF: 
        A class for encoding neural responses using time-delayed regression models. It includes 
        methods for cross-validation, hyperparameter search, and evaluation on test data.
        
    GpuTRF:
        A GPU-accelerated implementation of naplib's TRF model, enabling faster model fitting 
        and prediction by leveraging the GPU for computations.

Methods in TRF:
    __init__(model_name, dataset_obj):
        Initializes the TRF class with a given model name and dataset object.
        
    evaluate(trf_model, test_trial=None):
        Evaluates the trained TRF model on the test set, computing correlation between predicted 
        and actual neural responses.
        
    cross_validated_fit(tmax=50, tmin=0, num_folds=3, num_workers=1, use_nonlinearity=False):
        Performs cross-validated fitting of the TRF model to optimize the regularization parameter.
        
    grid_search_CV(lags=None, tmin=0, num_workers=1, num_folds=3, use_nonlinearity=False, test_trial=None, return_dict=False):
        Performs a grid search over possible lag values and cross-validation to find the optimal 
        regularization parameter and lag.

Methods in GpuTRF:
    __init__(tmin, tmax, sfreq, alpha=1):
        Initializes the GPU-accelerated TRF model with given time window and regularization parameter.
        
    fit(X, y):
        Fits the TRF model using time-delayed versions of the input features and neural responses.
        
    predict(X):
        Predicts the neural responses for the given input features using the fitted TRF model.
        
    score(X, y):
        Computes the R² score of the model's predictions.

        
Author: Bilal Ahmed
Date: 09-16-2024
Version: 1.0
License: MIT
Dependencies:
    - naplib
    - numpy (np)
    - cupy (cp)
    - sklearn.metrics (r2_score)
    - auditory_cortex.utils (for computing average test correlation)

Usage:
    This module is intended for neural encoding tasks, particularly for modeling temporal dynamics 
    between stimuli and neural responses. The TRF model is suitable for both CPU and GPU computations, 
    making it flexible for various computational settings.
"""

import gc
import numpy as np
import cupy as cp
import naplib as nl
from sklearn.metrics import r2_score

# local imports
from auditory_cortex import utils
import auditory_cortex.io_utils.io as io

import logging
logger = logging.getLogger(__name__)



class TRF:
    def __init__(self, model_name, dataset_assembler):
        """
        Args:
            num_freqs (int): Number of frequency channels on spectrogram
        """       
        self.model_name = model_name
        self.dataset_assembler = dataset_assembler
        logger.info(f"TRF object created for '{model_name}' model.")
        
    def evaluate(self, trf_model, n_test_trials=None, percent_duration=None):
        """Computes correlation on trials of test set for the model provided.
        
        Args:
            strf_model: naplib model = trained model
            n_test_trials: int = Number of random trial to be tested on. 
            percent_duration: float = Fraction of total test duration to use
                for evaluation, If None, use the entire test duration.
        Return:
            ndarray: (num_channels,)   
        """
        stim_ids, total_duration = self.dataset_assembler.dataloader.sample_stim_ids_by_duration(
            percent_duration=percent_duration, repeated=True, mVocs=self.dataset_assembler.mVocs
            )
        test_spect_list, all_test_spikes = self.dataset_assembler.get_testing_data(stim_ids=stim_ids)
        predicted_response = trf_model.predict(X=test_spect_list, n_offset=self.dataset_assembler.n_offset)
        predicted_response = np.concatenate(predicted_response, axis=0)     # gives (total_time, num_channels)
        all_test_spikes = np.concatenate(all_test_spikes, axis=1)           # gives (n_repeats, total_time, num_channels)
        
        corr = utils.compute_avg_test_corr(
            all_test_spikes, predicted_response, n_test_trials)
        return corr
    
    def get_mapping_set_ids(self, percent_duration=None, mVocs=False):
        """Returns random subset of stimulus ids, for the desired fraction of total 
        duration of training set as specified by percent_duration.
        
        Args:
            percent_duration: float = Fraction of total duration to consider.
                If None or >= 100, returns all stimulus ids.
            mVocs: bool = If True, mVocs trials are considered otherwise TIMIT
        
        Returns:
            list: stimulus subset for the fraction of duration.
        """
        stim_ids, stim_duration = self.dataset_assembler.dataloader.sample_stim_ids_by_duration(
            percent_duration, repeated=False, mVocs=mVocs
            )
        logger.info(f"Total duration={stim_duration:.2f} sec")
        return stim_ids


        # mapping_set = self.dataset_assembler.training_stim_ids
        # np.random.shuffle(mapping_set)
        # if N_sents is None or N_sents >= 100:	
        #     return mapping_set
        # else:
        #     required_duration = N_sents*10  # why is this 10?
        #     print(f"Mapping set for stim duration={required_duration:.2f} sec")
        #     stimili_duration=0
        #     for n in range(5, len(mapping_set)):
        #         stimili_duration += self.dataset_assembler.dataloader.get_stim_duration(
        #             mapping_set[n], mVocs=mVocs
        #             )
        #         if stimili_duration >= required_duration:
        #             break
        # return mapping_set[:n+1]

    def cross_validated_fit(
            self,
            tmax=50,
            tmin=0, 
            num_folds=3,
            mapping_set=None,
        ):
        """Computes score for the given lag (tmax) using cross-validated fit.
        
        Args:
            dataset:  = 
            tmax: int = lag (window width) in ms
            tmin: int = min lag start of window in ms
            num_workers: int = number of workers used by naplib.
            num_lmbdas: int  = number of regularization parameters (lmbdas)
            num_folds: int = number of folds of cross-validation
            use_nonlinearity: bool = using non-linearity with the linear model or not.
                Default = False.
        
        """
        tmin = tmin/1000
        tmax = tmax/1000
        sfreq = 1000/self.dataset_assembler.get_bin_width()
        num_channels = self.dataset_assembler.num_channels
        
        # Deprecated...
        if mapping_set is None:
            mapping_set = self.dataset_assembler.training_stim_ids
            np.random.shuffle(mapping_set)

        lmbdas = np.logspace(-5, 15, 21)
        lmbda_score = np.zeros(((len(lmbdas), num_channels)))
        size_of_chunk = int(len(mapping_set) / num_folds)

        for r in range(num_folds):
            logger.info(f"\n For fold={r}: ")
            if r<(num_folds-1):
                val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
            else:
                val_set = mapping_set[r*size_of_chunk:]
            train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

            train_x, train_y = self.dataset_assembler.get_training_data(stim_ids=train_set)
            val_x, val_y = self.dataset_assembler.get_training_data(stim_ids=val_set)
            for i, lmbda in enumerate(lmbdas):

                trf_model = GpuTRF(
                    tmin, tmax, sfreq, alpha=lmbda,
                    )
                trf_model.fit(X=train_x, y=train_y, n_offset=self.dataset_assembler.n_offset)

                # save validation score for lmbda..
                lmbda_score[i] += trf_model.score(X=val_x, y=val_y, n_offset=self.dataset_assembler.n_offset)

        lmbda_score /= num_folds
        max_lmbda_score = np.max(lmbda_score, axis=0)
        opt_lmbda = lmbdas[np.argmax(lmbda_score, axis=0)]
        
        # make sure to free up memory
        gc.collect()
        return max_lmbda_score, opt_lmbda

    def grid_search_CV(
            self,
            lag: int=None,      
            tmin: int= 0,
            num_folds: int= 3, 
            percent_duration=None,
        ):
        """Fits the linear model (with or without non-linearity) 
        by searching for optimal lag (max window lag) using cross-
        validation.

        Args:
            lag: int = lag (window width) in ms
            tmin: int = min lag start of window in ms
            num_folds: int = number of folds of cross-validation
            percent_duration: int = Percentage of training data (by duration) used to fit the model.
                For example, 50 means 50% of total training duration is used.
                Note: This applies only to the training set, not the test set.

        Return:
            corr: ndarray = (num_channels,) 
            optimal_lmbdas: ndarray = (num_channels,) 
            trf_model: GpuTRF = trained model object.

        """
        if lag is None:
            lag = 200

        mapping_set = self.get_mapping_set_ids(percent_duration=percent_duration, mVocs=self.dataset_assembler.mVocs)

        logger.info(f"\n Running for max lag={lag} ms")
        score, opt_lmbda = self.cross_validated_fit(
            tmax=lag,
            tmin=tmin, 
            num_folds=num_folds,
            mapping_set=mapping_set,
            )
            
        
        # get mapping data..
        mapping_x, mapping_y = self.dataset_assembler.get_training_data(mapping_set)

        sfreq = 1000/self.dataset_assembler.get_bin_width()
        tmax = lag/1000 # convert seconds to ms

        logger.info(f"Fitting model using optimal lag={lag} ms and optimal lmbda={opt_lmbda}")
        trf_model = GpuTRF(
                    tmin, tmax, sfreq, alpha=opt_lmbda,
                    )
        trf_model.fit(X=mapping_x, y=mapping_y, n_offset=self.dataset_assembler.n_offset)

        logger.info(f"Computing corr for test set...")
        corr = self.evaluate(trf_model)
        return corr, opt_lmbda, trf_model
    
    @staticmethod
    def load_saved_model(
            model_name, session, layer_ID, bin_width, shuffled=False,
            LPF=False, mVocs=False,
            tmax=300, tmin=0, dataset_name='ucsf'
        ):
        """Loads a saved weights and biases for the TRF model."""
        
        session = int(session)
        parameters = io.read_trf_parameters(
            model_name, session, bin_width=bin_width, 
            shuffled=shuffled, layer_ID=layer_ID, LPF=LPF, mVocs=mVocs,
            dataset_name=dataset_name, lag=tmax
            )
        # biases = io.read_trf_parameters(
        #         model_name, session, bin_width, shuffled,
        #         verbose=False, LPF=LPF, mVocs=mVocs,
        #         bias=True, dataset_name=dataset_name,
        #         lag=tmax
        #     )
        # alphas = io.read_alphas(
        #     model_name, session, bin_width=bin_width,
        #     shuffled=shuffled, layer_ID=layer_ID, LPF=LPF, mVocs=mVocs,
        #     dataset_name=dataset_name, lag=tmax
        #     )
        if parameters is None:
            # raise ValueError(f"Model parameters not found for session={session}")
            logger.warn(f"Model parameters not found for session={session}")
            return None
        
        # weights = weights[layer_ID]
        # biases = biases[layer_ID]
        
        tmax = tmax/1000
        sfreq = 1000/bin_width
        trf_model = GpuTRF(tmin, tmax, sfreq, alpha=parameters['alphas'])
        # trf_model.coef_ = (weights, biases)
        trf_model.coef_ = parameters['weights']
        trf_model.X_mean_ = parameters['x_mean']
        trf_model.X_std_ = parameters['x_std']
        trf_model.y_mean_ = parameters['y_mean']
        return trf_model
    
    @staticmethod
    def save_model_parameters(
            trf_model, model_name, layer_ID, session, bin_width, shuffled=False,
            LPF=False, mVocs=False, dataset_name='ucsf', tmax=200
            ):
        """Saves the weights and biases of the trained TRF model."""
        # weights, biases = trf_model.coef_
        # alphas = trf_model.alpha
        parameters = {
            'weights': trf_model.coef_,
            'alphas': trf_model.alpha,
            'x_mean': trf_model.X_mean_,
            'x_std': trf_model.X_std_,
            'y_mean': trf_model.y_mean_,
        }
        io.write_trf_parameters(
            model_name, session, parameters, bin_width=bin_width, 
            shuffled=shuffled, layer_ID=layer_ID, LPF=LPF, mVocs=mVocs,
            dataset_name=dataset_name, lag=tmax
            )
        # io.write_trf_parameters(
        #     model_name, session, biases, bin_width=bin_width, 
        #     shuffled=shuffled, layer_ID=layer_ID, LPF=LPF, mVocs=mVocs,
        #     bias=True, dataset_name=dataset_name, lag=tmax
        #     )
        # io.write_alphas(
        #     model_name, session, alphas, bin_width=bin_width,
        #     shuffled=shuffled, layer_ID=layer_ID, LPF=LPF, mVocs=mVocs,
        #     dataset_name=dataset_name, lag=tmax
        #     )
        

    def neural_prediction(
            self, model_name, session, layer_ID, bin_width, stim_ids,
                dataset_name, shuffled=False, mVocs=False, lag=200
            ):
        """Predicts the neural responses using the trained TRF model."""
        trf_model = self.load_saved_model(
            model_name, session, layer_ID, bin_width, shuffled=shuffled,
            LPF=False, mVocs=mVocs, dataset_name=dataset_name,
            tmax=lag
            )
            
        X, _ = self.dataset_assembler.get_testing_data(stim_ids)
        pred = trf_model.predict(X)
        return pred
    

class GpuTRF(nl.encoding.TRF):
    """GPU accelerated implementation of TRF model. 
    Built on top of naplib's TRF model.
    https://naplib-python.readthedocs.io/en/latest/references/encoding.html#trf 
    """
    def __init__(self, tmin, tmax, sfreq, alpha=0.1):
        """
        Args:
            tmin: int = start of time window in ms
            tmax: int = end of time window in ms
            sfreq: int = sampling frequency (Hz) of the data
            alpha: float or list = regularization parameter scalar or list of scalars.
                if list, fit separate model for channel of Y.
            
        """
        logger.info(f"GpuTRF object created with alpha={alpha}, tmin={tmin}, tmax={tmax}, sfreq={sfreq}")
        self.normalize_X = True
        self.center_y = True
        self.alpha = alpha
        if isinstance(alpha, float):
            # self.alpha = alpha
            self.n_alphas = 1
            self.model = LinearModel(alpha=alpha)
        elif isinstance(alpha, list) or (isinstance(alpha, np.ndarray) and alpha.ndim == 1):
            # self.alpha = alpha
            self.n_alphas = len(alpha)
            for i in range(self.n_alphas):
                self.models = [LinearModel(alpha=alp) for alp in alpha]
            # this is redundant, just to pass the first model to the parent class
            self.model = self.models[0]
        else:
            raise ValueError(f"Invalid alpha value={alpha}")
        super().__init__(
            tmin=tmin, tmax=tmax, sfreq=sfreq,
            estimator=self.model,
            n_jobs=1, show_progress=True
            )

    def fit(self, X, y, n_offset):
        """Given the input data, fits TRF model. Precisely, for each trial
        features in X, it's time delayed versions are stacked along features axis,
        and resultant ndarrays for each trial are concatenated along time axis.
        concatenates time delayed copies of X and fits the linear model.
        For example, if shape of trial features is (n_samples, n_features),
        then the features with time delays will be of shape (n_samples, n_features*n_lags).
        where n_lags is computed as (tmax-tmin)/sfreq.
        
        Args:
            X: list = list of ndarrays of shape (n_samples, n_features)
            y: list = list of ndarrays of shape (n_samples, n_targets)
        """
        self.ndim_y_ = y[0].ndim
        self.X_feats_ = X[0].shape[-1]
        self.n_targets_ = y[0].shape[1]
        self.n_models = None
        
        X_delayed, y_delayed = [], []
        for xx, yy in zip(X, y):
            X_tmp, y_tmp = self._delay_and_reshape(xx, yy)
            X_delayed.append(X_tmp[n_offset:])
            y_delayed.append(y_tmp)
        
        X_delayed = np.concatenate(X_delayed, axis=0)  # (samples, features)
        y_delayed = np.concatenate(y_delayed, axis=0)  # (samples, targets)

        # === Normalize features (per feature dimension) ===
        if getattr(self, "normalize_X", True):
            self.X_mean_ = X_delayed.mean(axis=0, keepdims=True)
            self.X_std_ = X_delayed.std(axis=0, keepdims=True) + 1e-6  # avoid div-by-zero
            X_delayed = (X_delayed - self.X_mean_) / self.X_std_

        # === Center target (optional) ===
        if getattr(self, "center_y", False):
            self.y_mean_ = y_delayed.mean(axis=0, keepdims=True)
            y_delayed = y_delayed - self.y_mean_
        else:
            self.y_mean_ = None

        if self.n_alphas == 1:
            self.model.fit(X_delayed, y_delayed)
        else:
            for i in range(self.n_alphas):
                self.models[i].fit(X_delayed, y_delayed[:,i])
        return self
    
    def predict(self, X, n_offset=0):
        """Predicts the response for the given input data. Hanldes time
        delays as explained in fit() method.
        
        Args:
            X: list = list of ndarrays of shape (n_samples, n_features)
        
        Return:
            list = list of ndarrays of shape (n_samples, n_targets)
        """
        if not hasattr(self, 'X_feats_'):
            raise ValueError(f'Must call .fit() before can call .predict()')

        X_delayed = []
        for xx in X:
            X_tmp,_ = self._delay_and_reshape(xx)
            # Normalize after delay-and-reshape using stored mean/std with matching shape
            # if hasattr(self, "X_mean_") and hasattr(self, "X_std_"):
            if getattr(self, "normalize_X", True):
                X_tmp = (X_tmp - self.X_mean_) / self.X_std_
            X_delayed.append(X_tmp[n_offset:])
            
        if self.n_alphas == 1:
            y_pred = []
            for xx in X_delayed:
                yp = self.model.predict(xx)
                if getattr(self, "center_y", False):
                    yp += self.y_mean_
                y_pred.append(yp)
        else:
            y_pred = [[] for _ in range(len(X_delayed))]
            for xi, xx in enumerate(X_delayed):
                for i in range(self.n_alphas):
                    # y_pred[xi].append(self.models[i].predict(xx))
                    yp = self.models[i].predict(xx)
                    if getattr(self, "center_y", False):
                        yp += self.y_mean_[:, i]
                    y_pred[xi].append(yp)
                    # y_pred.append()
                y_pred[xi] = np.stack(y_pred[xi], axis=1)
        return y_pred
    
    def score(self, X, y, n_offset=0):
        """Compute the coefficient of determination (score).
        """
        y = np.concatenate(y, axis=0)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        pred = self.predict(X, n_offset=n_offset)
        pred = np.concatenate(pred, axis=0)
        score = r2_score(y, pred, multioutput='raw_values')
        return score

    def normalize(self, X):
        return (X - np.mean(X, axis=0)[None,...])/np.std(X, axis=0)[None,...]
    
    @property
    def coef_(self):
        if not hasattr(self, 'ndim_y_'):
            raise ValueError(f'Must call fit() first before accessing coef_ attribute.')
        if hasattr(self, 'n_models') and self.n_models is not None:
            # for fitting multiple layers at the same time, this will almost never happen.
            return self.model.coef_.reshape(self.n_models, self.X_feats_, self._ndelays, self.n_targets_)
        else:
            if self.n_alphas ==1:	
                weights = self.model.coef_.reshape(self.X_feats_, self._ndelays, self.n_targets_)
                return weights
                # weights = self.model.coef_[0].reshape(self.X_feats_, self._ndelays, self.n_targets_)
                # biases = self.model.coef_[1]
                # return weights, biases
            else:
                coef = []
                biases = []
                for i in range(self.n_alphas):
                    coef.append(self.models[i].coef_.reshape(self.X_feats_, self._ndelays))
                return np.stack(coef, axis=-1)
                    # coef.append(self.models[i].coef_[0].reshape(self.X_feats_, self._ndelays))
                    # biases.append(self.models[i].coef_[1])
                # return np.stack(coef, axis=-1), np.stack(biases, axis=-1)
            
    @coef_.setter
    def coef_(self, value):
        """Sets the coefficients of the linear map and the bias term."""
        # Expecting value to be a tuple: (weights, bias)
        self.X_feats_ = value.shape[0]
        self.model.coef_ = value.reshape(-1, value.shape[-1])
        self.n_alphas = 1
        

        # weights, bias = value
        # self.model.coef_ = (weights, bias)
        # these attributes are needed to be able to predict.
        # self.n_alphas = 1
        # self.X_feats_ = weights.shape[0]
        

        

class LinearModel:
    """GPU accelerated linear model. Uses cupy for computations on GPU.
    It implements close form solution for linear regression with L2 regularization.
    """
    def __init__(self, alpha):
        """Create linear model with regularization parameter alpha."""
        self.alpha = alpha

    def fit(self, X, y):
        """Fit the linear model using the given data."""
        # X = self.adjust_for_bias(X)
        X = cp.array(X)
        y = cp.array(y)
        self.Beta = self.reg(X, y, lmbda=self.alpha)


    def predict(self, X):
        # X = self.adjust_for_bias(X)
        X = cp.array(X)
        pred = np.matmul(X, self.Beta)
        return cp.asnumpy(pred)
    
    
    def adjust_for_bias(self, X):
        """augment vector of 1's to X, which is the bias term."""
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    
    def reg(self, X,y, lmbda=0):
        """Fits linear regression parameters using the given data.
        Depending on the type of X and y, it uses numpy or cupy for computation.
        For linear model y = XB, it solves for B using the equation X^T X B = X^T y.
        
        Args:
            X (ndarray): (M,N) or (L,M,N) left-hand side array
            y (adarray): (M,) or (M,K) right-hand side array
            lmbda (float): regularization parameter (default=0)

        Returns:
            B (ndarray): (N,) or (N,K) or (L,N) or (L,N,K)
        """

        #check if incoming array is np or cp,
        #and decide which module to use...!
        if type(X).__module__ == np.__name__:
            module = np
        else:
            module = cp
        
        if X.ndim ==2:
            X = module.expand_dims(X,axis=0)
        d = X.shape[2]
        m = X.shape[1]

        # Create identity matrix and zero out the bias term (last diagonal entry)
        I = module.eye(d)
        # I[-1, -1] = 0  # Do not regularize the bias term

        A = module.matmul(X.transpose((0, 2, 1)), X) + m * lmbda * I
        B = module.matmul(X.transpose((0, 2, 1)), y)

        return module.linalg.solve(A, B).squeeze()
    
    @property
    def coef_(self):
        """Returns the coefficients of the linear map, excluding the bias term."""
        if not hasattr(self, 'Beta'):
            raise ValueError("Model has not been fit yet.")
        return self.Beta
        # return cp.asnumpy(self.Beta[:-1]), cp.asnumpy(self.Beta[-1])
    
    @coef_.setter
    def coef_(self, value):
        """Sets the coefficients of the linear map."""
        self.Beta = cp.asarray(value)

        # # Expecting value to be a tuple: (weights, bias)
        # weights, bias = value

        # # Ensure the input values are numpy/cupy arrays and assign them to the appropriate parts of Beta
        # weights = cp.asarray(weights)  # Convert weights to the appropriate format (using cupy here)
        # bias = cp.asarray(bias)  # Convert bias to the appropriate format (using cupy here)

        # # Concatenate the weights and bias to form the new Beta
        # self.Beta = np.concatenate([weights.reshape(-1, weights.shape[-1]), bias[None,:]], axis=0)	

