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


# import math
import numpy as np
# from scipy.signal import resample
from sklearn.metrics import r2_score
import gc
import naplib as nl
# from auditory_cortex import config
# from auditory_cortex.neural_data import dataset
# from auditory_cortex.dataloader import DataLoader
from auditory_cortex import utils
import auditory_cortex.io_utils.io as io
# from sklearn.linear_model import RidgeCV, ElasticNet, Ridge, PoissonRegressor
import cupy as cp
# from multiprocessing import Pool
# from tqdm.auto import tqdm
# import copy




class TRF:
	def __init__(self, model_name, dataset_obj):
		"""
		Args:
			num_freqs (int): Number of frequency channels on spectrogram
		"""       
		self.model_name = model_name
		self.dataset_obj = dataset_obj
		print(f"TRF object created for '{model_name}' model.")
		
	def evaluate(self, trf_model, test_trial=None):
		"""Computes correlation on trials of test set for the model provided.
		
		Args:
			strf_model: naplib model = trained model
			dataset:  = 
			test_trial: int = trial ID to be tested on. Default=None, 
				in which case, it tests on all the trials [0--10]
				and returns averaged correlations. 
			test_fewer_trials: bool = if True, test on randomly 
				choosen'test_trial' number of fewer trials
				instead of all 11 trials.
		Return:
			ndarray: (num_channels,)   
		"""

		test_spect_list, all_test_spikes = self.dataset_obj.get_testing_data()
		predicted_response = trf_model.predict(X=test_spect_list)
		predicted_response = np.concatenate(predicted_response, axis=0)
		
		corr = utils.compute_avg_test_corr(
			all_test_spikes, predicted_response, test_trial, mVocs=self.dataset_obj.mVocs)
		return corr
	
	def get_mapping_set_ids(self, N_sents=None, mVocs=False):
		"""Returns a smaller mapping set of given size."""
		mapping_set = self.dataset_obj.training_sent_ids
		np.random.shuffle(mapping_set)
		if N_sents is None or N_sents >= 100:	
			return mapping_set
		else:
			required_duration = N_sents*10
			print(f"Mapping set for stim duration={required_duration:.2f} sec")
			stimili_duration=0
			for n in range(5, len(mapping_set)):
				stimili_duration += self.dataset_obj.dataloader.get_stim_duration(
					mapping_set[n], mVocs=mVocs
					)
				if stimili_duration >= required_duration:
					break
		return mapping_set[:n+1]

	def cross_validated_fit(
			self,
			tmax=50,
			tmin=0, 
			num_folds=3,
			mapping_set=None,
			num_workers=1,
			use_nonlinearity=False,
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
		sfreq = 1000/self.dataset_obj.get_bin_width()
		num_channels = self.dataset_obj.num_channels
		
		# Deprecated...
		if mapping_set is None:
			mapping_set = self.dataset_obj.training_sent_ids
			np.random.shuffle(mapping_set)
		# mapping_set = self.dataset_obj.get_training_stim_ids()
		# lmbdas = np.logspace(-12, 7, 20)
		# lmbdas = np.logspace(-2, 5, 8)
		# lmbdas = np.logspace(-2, 10, 13)
		lmbdas = np.logspace(-5, 10, 16)
		lmbda_score = np.zeros(((len(lmbdas), num_channels)))
		size_of_chunk = int(len(mapping_set) / num_folds)

		for r in range(num_folds):
			print(f"\n For fold={r}: ")
			if r<(num_folds-1):
				val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
			else:
				val_set = mapping_set[r*size_of_chunk:]
			train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

			train_x, train_y = self.dataset_obj.get_training_data(stim_ids=train_set)
			val_x, val_y = self.dataset_obj.get_training_data(stim_ids=val_set)
			for i, lmbda in enumerate(lmbdas):

				# if use_nonlinearity:
				# 	estimator = PoissonRegressor(alpha=lmbda)
				# else:
				# 	estimator = Ridge(alpha=lmbda)
			
				# trf_model = nl.encoding.TRF(
				# 		tmin, tmax, sfreq, estimator=estimator,
				# 		n_jobs=num_workers, show_progress=True
				# 		)
				trf_model = GpuTRF(
					tmin, tmax, sfreq, alpha=lmbda,
					# n_jobs=1, show_progress=True
					)
				trf_model.fit(X=train_x, y=train_y)

				# save validation score for lmbda..
				lmbda_score[i] += trf_model.score(X=val_x, y=val_y)

		lmbda_score /= num_folds
		# avg_lmbda_score = np.mean(lmbda_score, axis=1)
		max_lmbda_score = np.max(lmbda_score, axis=0)
		opt_lmbda = lmbdas[np.argmax(lmbda_score, axis=0)]
		
		# make sure to free up memory
		gc.collect()
		return max_lmbda_score, opt_lmbda
	
		# avg_lmbda_score = np.mean(lmbda_score, axis=1)
		# max_lmbda_score = np.max(avg_lmbda_score)
		# opt_lmbda = lmbdas[np.argmax(avg_lmbda_score)]
		# return max_lmbda_score, opt_lmbda

	def grid_search_CV(
			self,
			lags: list = None,      
			tmin = 0,
			num_workers=1, 
			num_folds = 3, 
			use_nonlinearity = False, 
			test_trial=None,
			N_sents=None,
			return_dict=False
		):
		"""Fits the linear model (with or without non-linearity) 
		by searching for optimal lag (max window lag) using cross-
		validation.

		Args:
			dataset:  = 
			lags: list = lags (window width) in ms
			tmin: int = min lag start of window in ms
			num_workers: int = number of workers used by naplib.
			num_folds: int = number of folds of cross-validation
			use_nonlinearity: bool = using non-linearity with the linear model or not.
				Default = False.
			test_trial: int = trial ID to be tested on. Default=None, 
				in which case, it tests on all the trials [0--10]
				and returns averaged correlations. 
		Return:
			corr: ndarray = (num_channels,) 
			optimal_lags: ndarray = (num_channels,)
			optimal_lmbdas: ndarray = (num_channels,) 

		"""
		if lags is None:
			# lags = [50, 100, 150, 200, 250, 300, 350, 400]
			lags = [300]

		mapping_set = self.get_mapping_set_ids(N_sents=N_sents, mVocs=self.dataset_obj.mVocs)

		lag_scores = []
		opt_lmbdas = []
		for lag in lags:

			print(f"\n Running for max lag={lag} ms")
			score, opt_lmbda = self.cross_validated_fit(
				tmax=lag,
				tmin=tmin, 
				num_workers=num_workers,
				num_folds=num_folds,
				mapping_set=mapping_set,
				use_nonlinearity=use_nonlinearity
				)
			lag_scores.append(score)
			opt_lmbdas.append(opt_lmbda)
			
		# # modifying...
		# lag_scores = np.array(lag_scores)
		# opt_lmbdas = np.array(opt_lmbdas)

		# max_score_ind = np.argmax(lag_scores)
		# opt_lag = lags[max_score_ind]

		# ### for using RidgeCV ####
		# opt_lmbda = opt_lmbdas[max_score_ind]
		opt_lag = lags[0]
		opt_lmbda = opt_lmbdas[0]
		

		# get mapping data..
		mapping_x, mapping_y = self.dataset_obj.get_data(mapping_set)

		# # fit strf using opt lag and lmbda
		# if use_nonlinearity:
		# 	estimator = PoissonRegressor(alpha=opt_lmbda)
		# else:
		# 	estimator = Ridge(alpha=opt_lmbda)
		sfreq = 1000/self.dataset_obj.get_bin_width()
		tmax = opt_lag/1000 # convert seconds to ms

		# trf_model = nl.encoding.TRF(
		# 		tmin, tmax, sfreq, estimator=estimator,
		# 		n_jobs=num_workers, show_progress=True
		# 		)
		print(f"Fitting model using optimal lag={opt_lag} ms and optimal lmbda={opt_lmbda}")
		trf_model = GpuTRF(
					tmin, tmax, sfreq, alpha=opt_lmbda,
					# n_jobs=1, show_progress=True
					)
		trf_model.fit(X=mapping_x, y=mapping_y)

		print(f"Computing corr for test set...")
		corr = self.evaluate(trf_model, test_trial)
		return corr, opt_lag, opt_lmbda, trf_model
	
	def load_saved_model(
			self, model_name, session, layer_ID, bin_width, shuffled=False,
			tmax=300, tmin=0, 
		):
		"""Loads a saved weights and biases for the TRF model."""
		
		session = int(session)
		weights = io.read_trf_parameters(
				model_name, session, bin_width, shuffled,
				verbose=False, bias=False,
			)[layer_ID]

		biases = io.read_trf_parameters(
				model_name, session, bin_width, shuffled,
				verbose=False, bias=True,
			)[layer_ID]
		
		tmax = tmax/1000
		sfreq = 1000/bin_width
		trf_model = GpuTRF(tmin, tmax, sfreq, alpha=0.1)
		trf_model.coef_ = (weights, biases)
		return trf_model

	def neural_prediction(self, model_name, session, layer_ID, bin_width, stim_ids,
				shuffled=False):
		"""Predicts the neural responses using the trained TRF model."""
		trf_model = self.load_saved_model(
			model_name, session, layer_ID, bin_width, shuffled=shuffled
		)
		X, _ = self.dataset_obj.get_data(stim_ids)
		pred = trf_model.predict(X)
		return pred
	
	# @staticmethod
	# def compute_avg_test_corr(y_all_trials, y_pred, test_trial=None, mVocs=False):
	# 	"""Computes correlation for each trial and averages across all trials.
		
	# 	Args:
	# 		y_all_trials: (num_trials, num_bins)
	# 		y_pred: (num_bins,)
	# 		test_trial: int = integer in range=[0, 11], Default=None.
	# 			specifies number of trials to be tested on.

	# 	"""
	# 	trial_corr = []
	# 	if mVocs:
	# 		total_trial_repeats = 15
	# 	else:
	# 		total_trial_repeats = 11
	# 	# total_trial_repeats = y_all_trials.shape[0]
	# 	trial_ids = np.arange(total_trial_repeats)
	# 	if test_trial is not None:
	# 		np.random.shuffle(trial_ids)
	# 		trial_ids = trial_ids[:test_trial]
	# 	for tr in trial_ids:
	# 		trial_corr.append(cc_norm(y_all_trials[tr], y_pred))
	# 	trial_corr = np.stack(trial_corr, axis=0)
	# 	trial_corr = np.mean(trial_corr, axis=0)
	# 	return trial_corr

		# if test_trial is None:
		# 	for tr in range(total_trial_repeats):
		# 		trial_corr.append(cc_norm(y_all_trials[tr], y_pred))
		# 	trial_corr = np.stack(trial_corr, axis=0)
		# 	trial_corr = np.mean(trial_corr, axis=0)
		# else:
		# 	trial_corr = cc_norm(y_all_trials[test_trial], y_pred)
		# return trial_corr

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
		print(f"GpuTRF object created with alpha={alpha}, tmin={tmin}, tmax={tmax}, sfreq={sfreq}")
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

	def fit(self, X, y):
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
			X_delayed.append(X_tmp)
			y_delayed.append(y_tmp)
		
		X_delayed = np.concatenate(X_delayed, axis=0)
		y_delayed = np.concatenate(y_delayed, axis=0)

		if self.n_alphas == 1:
			self.model.fit(X_delayed, y_delayed)
		else:
			for i in range(self.n_alphas):
				self.models[i].fit(X_delayed, y_delayed[:,i])
		return self
	
	def predict(self, X):
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
			X_delayed.append(X_tmp)
			
		if self.n_alphas == 1:
			y_pred = []
			for xx in X_delayed:
				y_pred.append(self.model.predict(xx))
		else:
			y_pred = [[] for _ in range(len(X_delayed))]
			for xi, xx in enumerate(X_delayed):
				for i in range(self.n_alphas):
					y_pred[xi].append(self.models[i].predict(xx))
					# y_pred.append()
				y_pred[xi] = np.stack(y_pred[xi], axis=1)
		return y_pred
	
	def score(self, X, y):
		"""Compute the coefficient of determination (score).
		"""
	
		y = np.concatenate(y, axis=0)
		if y.ndim == 1:
			y = y[:, np.newaxis]

		pred = self.predict(X)
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
				weights = self.model.coef_[0].reshape(self.X_feats_, self._ndelays, self.n_targets_)
				biases = self.model.coef_[1]
				return weights, biases
			else:
				coef = []
				biases = []
				for i in range(self.n_alphas):
					coef.append(self.models[i].coef_[0].reshape(self.X_feats_, self._ndelays))
					biases.append(self.models[i].coef_[1])
				return np.stack(coef, axis=-1), np.stack(biases, axis=-1)
			
	@coef_.setter
	def coef_(self, value):
		"""Sets the coefficients of the linear map and the bias term."""
		# Expecting value to be a tuple: (weights, bias)
		weights, bias = value
		self.model.coef_ = (weights, bias)
		# these attributes are needed to be able to predict.
		self.n_alphas = 1
		self.X_feats_ = weights.shape[0]
		

		

class LinearModel:
	"""GPU accelerated linear model. Uses cupy for computations on GPU.
	It implements close form solution for linear regression with L2 regularization.
	"""
	def __init__(self, alpha):
		"""Create linear model with regularization parameter alpha."""
		self.alpha = alpha

	def fit(self, X, y):
		"""Fit the linear model using the given data."""
		X = self.adjust_for_bias(X)
		X = cp.array(X)
		y = cp.array(y)
		self.Beta = utils.reg(X, y, lmbda=self.alpha)

	def predict(self, X):
		X = self.adjust_for_bias(X)
		X = cp.array(X)
		pred = np.matmul(X, self.Beta)
		return cp.asnumpy(pred)
	
	
	def adjust_for_bias(self, X):
		"""augment vector of 1's to X, which is the bias term."""
		return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
	
	@property
	def coef_(self):
		"""Returns the coefficients of the linear map, excluding the bias term."""
		if not hasattr(self, 'Beta'):
			raise ValueError("Model has not been fit yet.")
		return cp.asnumpy(self.Beta[:-1]), cp.asnumpy(self.Beta[-1])
	
	@coef_.setter
	def coef_(self, value):
		"""Sets the coefficients of the linear map and the bias term."""
		# Expecting value to be a tuple: (weights, bias)
		weights, bias = value

		# Ensure the input values are numpy/cupy arrays and assign them to the appropriate parts of Beta
		weights = cp.asarray(weights)  # Convert weights to the appropriate format (using cupy here)
		bias = cp.asarray(bias)  # Convert bias to the appropriate format (using cupy here)

		# Concatenate the weights and bias to form the new Beta
		self.Beta = np.concatenate([weights.reshape(-1, weights.shape[-1]), bias[None,:]], axis=0)	

