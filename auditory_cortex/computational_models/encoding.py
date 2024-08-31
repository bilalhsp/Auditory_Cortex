import math
import numpy as np
from scipy.signal import resample

import naplib as nl
from auditory_cortex import config
# from auditory_cortex.neural_data import dataset
from auditory_cortex.dataloader import DataLoader
from auditory_cortex import utils
from sklearn.linear_model import RidgeCV, ElasticNet, Ridge, PoissonRegressor
import cupy as cp
from multiprocessing import Pool
from tqdm.auto import tqdm
import copy






class TRF:
	def __init__(self, model_name, dataset_obj):
		"""
		Args:
			num_freqs (int): Number of frequency channels on spectrogram
		"""       
		self.model_name = model_name
		self.dataset_obj = dataset_obj
		



	def evaluate(self, trf_model, test_trial=None):
		"""Computes correlation on trials of test set for the model provided.
		
		Args:
			strf_model: naplib model = trained model
			dataset:  = 
			test_trial: int = trial ID to be tested on. Default=None, 
				in which case, it tests on all the trials [0--10]
				and returns averaged correlations. 
		Return:
			ndarray: (num_channels,)   
		"""

		test_spect_list, all_test_spikes = self.dataset_obj.get_test_data()
		predicted_response = trf_model.predict(X=test_spect_list)
		predicted_response = np.concatenate(predicted_response, axis=0)

		corr = utils.compute_avg_test_corr(
			all_test_spikes, predicted_response, test_trial)
		return corr

	def cross_validted_fit(
			self,
			tmax=50,
			tmin=0, 
			num_folds=3,
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
		sfreq = 1000/self.dataset_obj.bin_width
		num_channels = self.dataset_obj.num_channels
		
		# Deprecated...
		mapping_set = self.dataset_obj.training_sent_ids
		# mapping_set = self.dataset_obj.get_training_stim_ids()
		# lmbdas = np.logspace(-12, 7, 20)
		# lmbdas = np.logspace(-2, 5, 8)
		# lmbdas = np.logspace(-2, 10, 13)
		lmbdas = np.logspace(-2, 12, 15)
		lmbda_score = np.zeros(((len(lmbdas), num_channels)))
		np.random.shuffle(mapping_set)
		size_of_chunk = int(len(mapping_set) / num_folds)

		for r in range(num_folds):
			print(f"\n For fold={r}: ")
			if r<(num_folds-1):
				val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
			else:
				val_set = mapping_set[r*size_of_chunk:]
			train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

			train_x, train_y = self.dataset_obj.get_data(stim_ids=train_set)
			val_x, val_y = self.dataset_obj.get_data(stim_ids=val_set)
			for i, lmbda in enumerate(lmbdas):

				if use_nonlinearity:
					estimator = PoissonRegressor(alpha=lmbda)
				else:
					estimator = Ridge(alpha=lmbda)
			
				trf_model = nl.encoding.TRF(
						tmin, tmax, sfreq, estimator=estimator,
						n_jobs=num_workers, show_progress=True
						)
				trf_model.fit(X=train_x, y=train_y)

				# save validation score for lmbda..
				lmbda_score[i] += trf_model.score(X=val_x, y=val_y)

		lmbda_score /= num_folds
		avg_lmbda_score = np.mean(lmbda_score, axis=1)
		max_lmbda_score = np.max(avg_lmbda_score)
		opt_lmbda = lmbdas[np.argmax(avg_lmbda_score)]
		return max_lmbda_score, opt_lmbda

	def grid_search_CV(
			self,
			lags: list = None,      
			tmin = 0,
			num_workers=1, 
			num_folds = 3, 
			use_nonlinearity = False, 
			test_trial=None,
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

		lag_scores = []
		opt_lmbdas = []
		for lag in lags:

			print(f"\n Running for max lag={lag} ms")
			score, opt_lmbda = self.cross_validted_fit(
				tmax=lag,
				tmin=tmin, 
				num_workers=num_workers,
				num_folds=num_folds,
				use_nonlinearity=use_nonlinearity
				)
			lag_scores.append(score)
			opt_lmbdas.append(opt_lmbda)

		max_score_ind = np.argmax(lag_scores)
		opt_lag = lags[max_score_ind]

		### for using RidgeCV ####
		opt_lmbda = opt_lmbdas[max_score_ind]

		# get mapping data..
		mapping_x, mapping_y = self.dataset_obj.get_data()

		# fit strf using opt lag and lmbda
		if use_nonlinearity:
			estimator = PoissonRegressor(alpha=opt_lmbda)
		else:
			estimator = Ridge(alpha=opt_lmbda)
		sfreq = 1000/self.dataset_obj.bin_width
		tmax = opt_lag/1000 # convert seconds to ms
		trf_model = nl.encoding.TRF(
				tmin, tmax, sfreq, estimator=estimator,
				n_jobs=num_workers, show_progress=True
				)
		trf_model.fit(X=mapping_x, y=mapping_y)

		corr = self.evaluate(trf_model, test_trial)
		return corr, opt_lag, opt_lmbda
	
class GpuTRF(nl.encoding.TRF):
	"""GPU accelerated TRF model."""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self.model = LinearModel(alpha=2)

	def fit(self, X, y):

		# gpu implementation
		# concatenate along 2nd last axis (the time axis),
		X = np.concatenate(X, axis=-2) 	
		y = np.concatenate(y, axis=0)

		if y.ndim == 1:
			y = y[:, np.newaxis]
		#
		self.ndim_y_ = y.ndim
		self.X_feats_ = X.shape[-1]
		self.n_targets_ = y.shape[1]
		self.n_models = None
		# self.y_feats_ = 1


		print(self.n_models)
		print(f"X shape: {X.shape}")
		print(f"y shape: {y.shape}")
		
		# Delay inputs and reshape
		if X.ndim == 3:
			# if there is layer axis in X, then we need to delay each layer separately
			self.n_models = X.shape[0]
			X_delayed = []
			for i in range(X.shape[0]):
				x_tmp, _ = self._delay_and_reshape(X[i])
				X_delayed.append(x_tmp)
			X_delayed = np.concatenate(X_delayed, axis=0)
		else:
			X_delayed, _ = self._delay_and_reshape(X)
		y_delayed = y
		
		print(f"X shape: {X_delayed.shape}")
		print(f"y shape: {y_delayed.shape}")

		self.model.fit(X_delayed, y_delayed)

		return self
	
	def predict(self, X):
		if not hasattr(self, 'X_feats_'):
			raise ValueError(f'Must call .fit() before can call .predict()')
		
		X = np.concatenate(X, axis=-2) 	

		# Delay inputs and reshape
		if X.ndim == 3:
			# if there is layer axis in X, then we need to delay each layer separately
			# self.n_models = X.shape[0]
			X_delayed = []
			for i in range(X.shape[0]):
				x_tmp, _ = self._delay_and_reshape(X[i])
				X_delayed.append(x_tmp)
			X_delayed = np.concatenate(X_delayed, axis=0)
		else:
			X_delayed, _ = self._delay_and_reshape(X)
		
		y_pred = self.model.predict(X_delayed)
		return y_pred
	
	def score(self, X, y):
		"""Compute the negative mean squared error of the model on the given data.
		"""
		X = np.concatenate(X, axis=-2) 	
		y = np.concatenate(y, axis=0)

		if y.ndim == 1:
			y = y[:, np.newaxis]

		# Delay inputs and reshape
		if X.ndim == 3:
			# if there is layer axis in X, then we need to delay each layer separately
			X_delayed = []
			for i in range(X.shape[0]):
				x_tmp, _ = self._delay_and_reshape(X[i])
				X_delayed.append(x_tmp)
			X_delayed = np.concatenate(X_delayed, axis=0)
		else:
			X_delayed, _ = self._delay_and_reshape(X)

		score = self.model.score(X_delayed, y)

		return score
	
	@property
	def coef_(self):
		if not hasattr(self, 'ndim_y_'):
			raise ValueError(f'Must call fit() first before accessing coef_ attribute.')
		if hasattr(self, 'n_models') and self.n_models is not None:
			return self.model.coef_.reshape(self.n_models, self.X_feats_, self._ndelays, self.n_targets_)
		else:
			return self.model.coef_.reshape(self.X_feats_, self._ndelays, self.n_targets_)
		

class LinearModel:
	"""GPU accelerated linear model."""
	def __init__(self, alpha):
		"""Create linear model with regularization parameter alpha."""
		self.alpha = alpha

	def fit(self, X, y):
		X = cp.array(X)
		y = cp.array(y)
		self.Beta = utils.reg(X, y, lmbda=self.alpha)
		# self.model.fit(X, y)

	def predict(self, X):
		X = cp.array(X)
		pred = np.matmul(X, self.Beta)
		return cp.asnumpy(pred)
	
	def score(self, X, y):
		pred = self.predict(X)
		loss = -1*utils.mse_loss(y, pred)
		return loss
	
	@property
	def coef_(self):
		if not hasattr(self, 'Beta'):
			raise ValueError("Model has not been fit yet.")
		return cp.asnumpy(self.Beta)
	




# # Old implementations

# class STRF:
#     def __init__(self, session, estimator=None, num_workers=1, num_freqs=80, 
#             tmin=0, tmax = 0.3, bin_width=20, train_dataset_size=0.8):
#         """
#         Args:
#             num_freqs (int): Number of frequency channels on spectrogram
#             tmin: receptive field begins at (ms)
#             tmax: receptive field ends at (ms)
#             sfreq: sampling frequency of data (Hz)

#             train_dataset_size: [0.0, 1.0]= fraction of total dataset 
#             to be used as training/val split. Defalt=0.8
#                 (excluding the test split only)
#         """
#         self.model_name = 'strf_model'
#         session = str(int(session))
#         self.bin_width = bin_width
#         data_dir = config['neural_data_dir']
#         # self.dataset = NeuralData(data_dir, session)
#         # self.dataset.extract_spikes(bin_width=self.bin_width, delay=0)
#         # self.fs = self.dataset.fs
#         self.dataloader = DataLoader()
#         self.session_spikes = self.dataloader.get_session_spikes(
#             session=session, bin_width=bin_width, delay=0
#             )
#         self.fs = self.dataloader.metadata.get_sampling_rate()
#         self.num_freqs = num_freqs # num_freqs in the spectrogram

#         if estimator is None:
#             alphas = np.logspace(-2, 5, 6)
#             estimator = RidgeCV(alphas=alphas, cv=5)


#         # creating a STRF model...
#         sfreq = 1000/bin_width # 50 (since bin_width is in ms)
#         self.strf_model = nl.encoding.TRF(
#             tmin, tmax, sfreq, estimator=estimator,
#             n_jobs=num_workers, show_progress=True
#             )

#         # sents = np.arange(1,499)
#         # self.random_sent_ids = np.random.permutation(sents)
#         # self.size_training_dataset = int(train_dataset_size*(sents.size))
#         sent_IDs = self.dataloader.sent_IDs
#         self.testing_sent_ids = self.dataloader.test_sent_IDs
#         self.training_sent_ids = sent_IDs[np.isin(sent_IDs, self.testing_sent_ids, invert=True)]

#     def get_sample(self, sent, third=None):

#         # spikes = self.dataset.unroll_spikes([sent], third=third).astype(np.float32)
#         # aud = self.dataset.audio(sent)
	
#         spikes = self.session_spikes[sent]
#         num_bins = spikes[0]
#         aud = self.dataloader.metadata.stim_audio(sent)
#         # Getting the spectrogram at 10 ms and then resample to match the bin_width
#         spect = nl.features.auditory_spectrogram(aud, self.fs, frame_len=10)

#         if third is not None:
#             # bin_width = self.bin_width/1000.0
#             # n = int(np.ceil(round(self.dataset.duration(sent)/bin_width, 3)))
			
#             # # store boundaries of sent thirds...
#             # one_third = int(n/3)
#             # two_third = int(2*n/3)
#             # sent_sections = [0, one_third, two_third, n]

#             # spect = resample(spect, n, axis=0)
#             # # print(spect.shape)
#             # spect = spect[sent_sections[third-1]: sent_sections[third]]
#             # print(spect.shape)
#             ...
			
#         else:
#             spect = resample(spect, num_bins, axis=0)
#         # read spikes for the sent id, as (time, channel)
#         # spikes = self.spikes[sent].astype(np.float32)

#         # get spectrogram for the audio inputs, as (time, freq) 
#         # spect = nl.features.auditory_spectrogram(aud, self.fs)
#         # resample spect to get same # number of time samples as in spikes..
#         # spect = resample(spect, spikes.shape[0], axis=0)
#         # spect = resample(spect, samples, axis=0)

#         # spect has 128 channels at this point, we can reduce the channels 
#         # for easy training, to match speech2text spectrogram 80 for now...
#         spect = resample(spect, self.num_freqs, axis=1)
#         return spect, spikes#np.expand_dims(spikes[:,ch], axis=1)


	

		

	
#     def fit(self, third=None):
#         """

#         """
#         # training_sent_ids = self.random_sent_ids[:self.size_training_dataset]
		
#         # collecting data after pre-processing...
#         train_spect_list = []
#         train_spikes_list = []
#         for sent in self.training_sent_ids:
#             spect, spikes = self.get_sample(sent, third=third)
#             train_spect_list.append(spect)
#             train_spikes_list.append(spikes)
		
#         self.strf_model.fit(X=train_spect_list, y=train_spikes_list)
#         corr = self.evaluate(third=third)
#         return corr
	
#     def evaluate(self, third=None):
#         # testing_sent_ids = self.random_sent_ids[self.size_training_dataset:]
	
#         # collecting data after pre-processing...
#         test_spect_list = []
#         test_spikes_list = []
#         for sent in self.testing_sent_ids:
#             spect, spikes = self.get_sample(sent, third=third)
#             test_spect_list.append(spect)
#             test_spikes_list.append(spikes)
		
#         corr = self.strf_model.corr(X=test_spect_list, y=test_spikes_list)
#         return corr
	
#     def get_coefficients(self):
#         """Returns the coefficients of the linear map from STRF to 
#         neural responses.
		
#         Returns:
#             ndarray = strf_model coefficients (num_ch, n_features_X, n_lags)
#         """
#         return self.strf_model.coef_




