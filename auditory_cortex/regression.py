import numpy as np
import torch
import os
from scipy import linalg, signal
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor,Wav2Vec2Processor, Wav2Vec2ForCTC

from auditory_cortex.dataset import Neural_Data
from auditory_cortex.feature_extractors import Feature_Extractor_S2T,Feature_Extractor_GRU,FeatureExtractorW2L
import auditory_cortex.utils as utils

#from sklearn.decomposition import PCA
# import rnn_model.speech_recognition as speech_recognition
import matplotlib.pyplot as plt
import torchaudio

class transformer_regression():
  def __init__(self, dir, subject, model='speech2text', load_features = True):
    self.dir = os.path.join(dir, subject)
    print("Creating dataset and other objects...")
    self.dataset = Neural_Data(dir, subject)
    if model == 'speech2text':
        print(f"Creating regression obj for: 'speech2text'")
        self.model_name=model
        self.layers = ["model.encoder.conv.conv_layers.0","model.encoder.conv.conv_layers.1",
                        "model.encoder.layers.0.fc2","model.encoder.layers.1.fc2",
                        "model.encoder.layers.2.fc2","model.encoder.layers.3.fc2",
                        "model.encoder.layers.4.fc2","model.encoder.layers.5.fc2",
                        "model.encoder.layers.6.fc2","model.encoder.layers.7.fc2",
                        "model.encoder.layers.8.fc2","model.encoder.layers.9.fc2",
                        ]
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.model_extractor = Feature_Extractor_S2T(self.model, self.processor, self.layers)

    elif model == 'wav2vec':
        print(f"Creating regression obj for: 'wav2vec'")
        self.model_name = model
        self.layers = ['wav2vec2.feature_extractor.conv_layers.0.conv','wav2vec2.feature_extractor.conv_layers.1.conv',
                       'wav2vec2.feature_extractor.conv_layers.2.conv','wav2vec2.feature_extractor.conv_layers.3.conv',
                       'wav2vec2.feature_extractor.conv_layers.4.conv','wav2vec2.feature_extractor.conv_layers.5.conv',
                       'wav2vec2.feature_extractor.conv_layers.6.conv']
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.model_extractor = feature_extractor_wav2vec(self.model, self.processor, self.layers)
        self.seq_lengths = {s:int(np.floor(self.dataset.duration(s)/0.02 - 0.25)) for s in np.arange(1,499)}
    
    elif model == 'gru':
        print(f"Creating regression obj for: 'gru'")
        self.model_name = model
        self.layers = ['birnn_layers.0.BiGRU','birnn_layers.1.BiGRU','birnn_layers.2.BiGRU','birnn_layers.3.BiGRU','birnn_layers.4.BiGRU']
        self.model = speech_recognition.SpeechRecognitionModel(3,5,512,29,128,2,0.1)
        path = os.path.join(dir, 'rnn_model')
        weights_file = "epoch_250.pt"
        checkpoint = torch.load(os.path.join(path,weights_file),map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model_extractor = Feature_Extractor_GRU(self.model, self.layers)
        self.spect = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

    else:
         # This block for wav2letter trained model...!
         print(f"Creating regression obj for: '{model.model_name}'")
         self.model = model
         self.model_extractor = FeatureExtractorW2L(self.model)
         self.model_name = model.model_name
         self.layers = self.model_extractor.layers

    if load_features:
        # print("Loading model features now...!")
        # self.load_features()
        self.raw_features = self.extract_features()

  # def load_features(self, resample=False, bin_width=5, sents=np.arange(1,499)):
  #   """
  #   | wrapper function to load features dict into
  #   | 'self.features'
  #   """
  #   print("Loading model features now...!")
  #   # self.features = self.get_transformer_features(sents)
  #   self.raw_features = self.extract_features(sents)
  #   if resample:
  #     features = self.resample(features, bin_width)
  #   else:
  #     features = self.raw_features
  #   self.features = self.unroll_time(features)
    

  def get_transformer_features(self, sents=np.arange(1,499)):
    """
    | Returns features (as dict) for given 'sents'
    | and all layers.
    """
    features = [{} for _ in range(len(self.layers))]
    demean_features = [{} for _ in range(len(self.layers))]
    f_mean = {}    
    feats = {}
    demean_feats = {}
    # sum = 0
    for x, i in enumerate(sents):
      self.model_extractor.translate(self.dataset.audio(i))

      for j, l in enumerate(self.layers):
        features[j][x] = self.model_extractor.features[l]
        # a = self.dataset.duration(i)
        # b = self.model_extractor.features[l].shape[0]
        # w = 20
        # offset = 0
        # print(f"Duration: {a:.3f}, features: {b}, bio: {a*1000/w + offset},diff:{b - np.floor(a*1000/w + offset)}")
        # # sum += b - round(a*1000/19.9)
        if self.model_name=='wav2vec':
                features[j][x] = features[j][x][:self.seq_lengths[i]]
        #f_mean[x] = np.mean(features[j][x], axis = 0)    
        #demean_features[j][x]= features[j][x] - f_mean[x]
  
    for j, l in enumerate(self.layers):
      feats[j] = np.concatenate([features[j][i] for i,se in enumerate(sents)], axis=0)
      #demean_feats[j] = np.concatenate([demean_features[j][i] for i,se in enumerate(sents)], axis=0)
    return feats#, demean_feats


    ##############################
    ## New functions...
    ##############################
  def extract_features(self, sents=np.arange(1,499)):
    """
    Returns all layer features for given 'sents'
    
    Args:
        sents (list, optional): List of sentence ID's to get the features for. 

    Returns:
        List of dict: List index corresponds to layer number carrying 
                      dict of extracted features for all sentences. 
    """
    features = [{} for _ in range(len(self.layers))]
    for x, i in enumerate(sents):
      self.model_extractor.translate(self.dataset.audio(i))
      for j, l in enumerate(self.layers):
        features[j][i] = self.model_extractor.features[l]
        if self.model_name=='wav2vec':
                features[j][i] = features[j][x][:self.seq_lengths[i]]
    return features

  def resample(self, features, bin_width):
    """
    resample all layer features to specific bin_width

    Args:
        bin_width (float): width of data samples in ms (1000/sampling_rate).

    Returns:
        List of dict: all layer features (resampled at required sampling_rate).
    """
    resampled_features = [{} for _ in range(len(self.layers))]
    bin_width = bin_width/1000 # ms
    for sent in features[0].keys():
      n = int(np.ceil(round(self.dataset.duration(sent)/bin_width, 3)))
      for j, l in enumerate(self.layers):
        resampled_features[j][sent] = signal.resample(features[j][sent],n, axis=0)
    return resampled_features

  def unroll_time(self, features):
    """
    Unroll and concatenate time axis of extracted features.

    Args:
        features (List of dict): features for all layers.
    
    Returns:
        dict: 
    """
    feats = {}
    for j, l in enumerate(self.layers):
      feats[j] = np.concatenate([features[j][sent] for sent in features[j].keys()], axis=0)
    return feats

  def load_features_and_spikes(self, bin_width=5, delay=0, offset=0, sents = np.arange(1,499), load_raw=False):
    if load_raw:
      self.raw_features = self.extract_features(sents)
    self.features = self.unroll_time(self.resample(self.raw_features, bin_width))
    self.spikes = self.all_channel_spikes(bin_width=bin_width, delay=delay, offset=offset, sents=sents)

  def corr_coeffs(self, layers=None, channels=None):
    if layers == None:
      layers = np.arange(len(self.layers))
    if channels == None:
      channels = np.arange(self.dataset.num_channels)
    train_cc = np.zeros((len(layers), len(channels)))
    val_cc = np.zeros((len(layers), len(channels)))
    test_cc = np.zeros((len(layers), len(channels)))
    for l,layer in enumerate(layers):
      print(f"Computing correlations for layer {layer}")
      for c,ch in enumerate(channels):
        train_cc[l,c], val_cc[l,c], test_cc[l,c] = \
                self.compute_cc_norm(self.features[layer], self.spikes[ch])
    return train_cc, val_cc, test_cc

  def save_corr_coeffs(self, win, delay, file_path):
    print(f"Working on win: {win}ms, delay: {delay}ms")
    self.load_features_and_spikes(bin_width=win, delay=delay)
    train_cc, val_cc, test_cc = self.corr_coeffs()
    corr = {'train': train_cc, 'val': val_cc, 'test': test_cc, 'win': win, 'delay': delay}
    data = utils.write_to_disk(corr, file_path)



#########################################    ##############################



  def simply_spikes(self, sent_s=1, sent_e=499, ch=0, delay=0, def_w=40, offset=0):
    spikes ={}
    for x,i in enumerate(range(sent_s,sent_e)):
      spikes[x] = torch.tensor(self.dataset.retrieve_spike_counts(sent=i,win=def_w,delay=delay,early_spikes=False,
                                                                  offset=offset)[ch])
    spikes = torch.cat([spikes[i] for i in range(sent_e - sent_s)], dim = 0).numpy()
    return spikes

  def all_channel_spikes(self, bin_width=40, delay=0, offset=0, sents = np.arange(1,499)):
    spikes = []
    result = {}
    for x,i in enumerate(sents):
        spikes.append(self.dataset.retrieve_spike_counts(sent=i,win=bin_width,delay=delay,early_spikes=False,offset=offset))
    for ch in range(self.dataset.num_channels):
        result[ch] = np.concatenate([spikes[i][ch] for i in range(len(spikes))], axis=0)

    return result

  def get_cc_norm_layer(self, layer, win, delay=0, sents= np.arange(1,499),normalize = False, load_features=False):
    """
    | Gives correlation coefficients for given
    | 'layer' (and all channels)
    """
    print(f"Computing correlations for layer:{layer} ...")
    sp = 1
    num_channels = self.dataset.num_channels
    train_cc_norm = np.zeros(num_channels)
    val_cc_norm = np.zeros(num_channels)
    test_cc_norm = np.zeros(num_channels)
    
    feats, spikes = self.get_feats_and_spikes(layer, win, delay, sents, load_features)
    if normalize:
        sp_all_channels = self.dataset.signal_power(win)
    for ch in range(num_channels):
        if normalize:
            sp = sp_all_channels[ch]
        train_cc_norm[ch], val_cc_norm[ch], test_cc_norm[ch] = self.compute_cc_norm(feats, spikes[ch], sp, normalize=normalize)
        
    return train_cc_norm, val_cc_norm, test_cc_norm
  
  def get_feats_and_spikes(self, layer, win, delay=0, sents= np.arange(1,499), load_features=False):
    """
    | Gives features and spikes data for given
    | 'layer' and all channels.
    """
    if load_features:
        print("Loading model layer features now...!")
        self.load_features()
    
    def_w, offset = self.model_extractor.def_bin_width(layer)            
    k = int(win/def_w)    # 40 is the min, bin size for 'Speech2Text' transformer model 
    feats = self.features[layer]
    spikes = self.all_channel_spikes(sents=sents, delay=delay, bin_width=def_w, offset=offset)
    if k>1:
      feats = utils.down_sample(feats, k)
      for ch in range(self.dataset.num_channels):
          spikes[ch] = utils.down_sample(spikes[ch],k)

    return feats, spikes

  def get_cc_norm(self, layer, win, channel, delay=0, normalize = False, sents= np.arange(1,499), load_features=False):
    """
    | Gives correlation coefficient for given 
    | 'layer' and 'channel' 
    """
    if load_features:
        print("Loading model layer features now...!")
        self.load_features()
    def_w, offset = self.model_extractor.def_bin_width(layer)       
    k = int(win/def_w)    # 40 is the min, bin size for 'Speech2Text' transformer model 
    feats = self.features[layer]   
    y = self.all_channel_spikes(sents=sents, delay=delay, bin_width=def_w, offset=offset)[channel]
    if k>1:
      feats = utils.down_sample(feats, k)
      y = utils.down_sample(y,k)
    if normalize:
        sp = self.dataset.signal_power(win)[channel]
    else:
        sp = 1
    r2t, r2v,r2tt = self.compute_cc_norm(feats, y, sp, normalize=normalize)
    return r2t, r2v,r2tt

  def compute_cc_norm(self, x, y, sp=1, normalize=False):
    """
    | return correlation coefficient for given 
    | data (x,y), and optional 'sp' and 'normalize' flag.
    """
    # provide 'sp' for normalized correlation coefficient...!
    r2t = np.zeros(1)
    r2v = np.zeros(1)
    r2tt = np.zeros(1)
    
    m = int(x.shape[0])
    n2 = int(m*0.9)
    x_test = x[n2:, :]
    y_test = y[n2:]    
    
    # signal power, will be used for normalization
    #sp = self.dataset.signal_power(win, channel)
    for i in range(5):
        a = int(i*0.2*n2)
        b = int((i+1)*0.2*n2)
  
        x_val = x[a:b, :] 
        y_val = y[a:b] 
        
        x_train = np.concatenate((x[:a,:], x[b:n2,:]), axis=0)
        y_train = np.concatenate((y[:a], y[b:n2]))
        
        # Linear Regression...!
        B = self.regression_param(x_train, y_train)
        y_hat_train = self.predict(x_train, B)
        y_hat_val = self.predict(x_val, B)
        y_hat_test = self.predict(x_test, B)
        
        #Normalized correlation coefficient
        r2t += self.cc_norm(y_hat_train, y_train, sp, normalize=normalize)
        r2v += self.cc_norm(y_hat_val, y_val, sp, normalize=normalize)
        r2tt += self.cc_norm(y_hat_test, y_test, sp, normalize=normalize)
        
    r2t /= 5
    r2v /= 5
    r2tt /= 5
   
    return r2t, r2v,r2tt  

  def compute_and_store_corr(self, wins, delays, file_path):
    """computes correlations for all layers and channels,
    | for all combinations of 'wins' and 'delays' and stores them 
    | to the 'file_path'.
    | wins: list
    | delays: list
    | file_path: path of csv file
    """
    num_layers = len(self.layers)
    num_channels = self.dataset.num_channels
    for win in wins:
        for delay in delays:
            train_cc = np.zeros((num_layers, num_channels))
            val_cc = np.zeros((num_layers, num_channels))
            test_cc = np.zeros((num_layers, num_channels)) 
            for layer in range(0, num_layers):
                train_cc[layer,:], val_cc[layer,:], test_cc[layer,:] = self.get_cc_norm_layer(layer, win, delay, normalize=False)
            corr = {'train': train_cc, 'val': val_cc, 'test': test_cc, 'win': win, 'delay': delay}
            data = utils.write_to_disk(corr, file_path)

  def get_Poiss_scores_layer(self, layer, win, delay=0, sents= np.arange(1,499), load_features=False):
    print(f"Computing Poisson scores for layer:{layer} ...")
    num_channels = self.dataset.num_channels
    train_scores = np.zeros(num_channels)
    val_scores = np.zeros(num_channels)
    test_scores = np.zeros(num_channels)
    
    feats, spikes = self.get_feats_and_spikes(layer, win, delay, sents, load_features)
    for ch in range(num_channels):
        train_scores[ch], val_scores[ch], test_scores[ch] = self.compute_poiss_scores(feats, spikes[ch])
        
    return train_scores, val_scores, test_scores  

  def compute_poiss_scores(self, x, y):
    # provide 'sp' for normalized correlation coefficient...!
    ps_t = np.zeros(1)
    ps_v = np.zeros(1)
    ps_tt = np.zeros(1)
    
    m = int(x.shape[0])
    n2 = int(m*0.9)
    x_test = x[n2:, :]
    y_test = y[n2:]    
    
    # signal power, will be used for normalization
    #sp = self.dataset.signal_power(win, channel)
    for i in range(5):
        a = int(i*0.2*n2)
        b = int((i+1)*0.2*n2)
        
        x_val = x[a:b, :] 
        y_val = y[a:b] 
        
        x_train = np.concatenate((x[:a,:], x[b:n2,:]), axis=0)
        y_train = np.concatenate((y[:a], y[b:n2]))
        
        # Poisson Regression...!
        
        poiss_model = utils.poiss_regression(x_train, y_train)


        
        #Poisson Scores
        ps_t += utils.poisson_regression_score(poiss_model, x_train, y_train)
        ps_v += utils.poisson_regression_score(poiss_model, x_val, y_val)
        ps_tt += utils.poisson_regression_score(poiss_model, x_test, y_test)
        
    ps_t /= 5
    ps_v /= 5
    ps_tt /= 5
   
    return r2t, r2v,r2tt
 
  def cc_norm(self, y_hat, y, sp, normalize=False):
    # if 'normalize' = True, use signal power as factor otherwise use normalize CC formula i.e. 'un-normalized'
    if normalize:
        factor = sp
    else:
        factor = np.var(y)  

    return np.cov(y_hat, y)[0,1]/(np.sqrt(np.var(y_hat)*factor))

  def regression_param(self, X, y):
    B = linalg.lstsq(X, y)[0]
    return B

  def predict(self, X, B):
    return X@B

  def r2(self, labels, predictions):
    score = 0.0
    mean = np.mean(labels)
    denom = np.sum(np.square(labels - mean))
    num = np.sum(np.square(labels - predictions))
    score = 1 - num/denom
    return score
  def regression_score(self, X,y, B):
    y_hat = self.predict(X,B)
    return self.r2(y, y_hat)


################################################################################################
##############Redundant functions....!
#############################################################


  def demean_spikes(self, sent_s=1, sent_e=499, ch=0, w = 40):
    spikes ={}
    spk_mean = {}
    for x,i in enumerate(range(sent_s,sent_e)):
      spikes[x] = torch.tensor(self.dataset.retrieve_spike_counts(sent=i, win=w ,early_spikes=False)[ch])
      spk_mean[x] = torch.mean(spikes[x], dim = 0)
      spikes[x] = spikes[x] - spk_mean[x]
    spikes = torch.cat([spikes[i] for i in range(sent_e - sent_s)], dim = 0).numpy()
    return spikes



  def benchmark_r2_score(self, w = 40, sent = 12):
    #These sentences have repeated trials...!
    #sents = [12,13,32,43,56,163,212,218,287,308]
    r2_scores = np.zeros(self.dataset.num_channels)
    #trials = obj.dataset.get_trials(13)
    spkk = self.dataset.retrieve_spike_counts_for_all_trials(sent=sent, w=w)
  
    for i in range(self.dataset.num_channels):
      h1 = np.mean(spkk[i][0:6], axis=0)
      h2 = np.mean(spkk[i][6:], axis=0)
      r2_scores[i] = self.r2(h1,h2)
    return r2_scores


  def compute_r2(self, layer, win):
    k = int(win/40)    # 40 is the min, bin size for 'Speech2Text' transformer model 
    # print(f"k = {k}")
    r2t = np.zeros(self.dataset.num_channels)
    r2v = np.zeros(self.dataset.num_channels)
    pct = np.zeros(self.dataset.num_channels)
    pcv = np.zeros(self.dataset.num_channels)

    #downsamples if k>1 
    if k >1:
      feats = utils.down_sample(self.features[layer], k)
    else:
      feats = self.features[layer]

    m = int(feats.shape[0] *0.75)
    x_train = feats[0:m, :]
    x_test = feats[m:, :]
    
    for i in range(self.dataset.num_channels):
      y = self.simply_spikes(ch=i)
      if k>1:
        y = utils.down_sample(y,k)
      y_train = y[0:m]
      y_test = y[m:]
      B = self.regression_param(x_train, y_train)
 
      r2t[i] = self.regression_score(x_train, y_train, B)
      r2v[i] = self.regression_score(x_test, y_test, B)
      pct[i] = (np.corrcoef(self.predict(x_train, B), y_train)[0,1])**2
      pcv[i] = (np.corrcoef(self.predict(x_test, B), y_test)[0,1])**2
    return r2t, r2v, pct, pcv



  def compute_r2_channel(self, layer, win, channel, delay):
    k = int(win/40)    # 40 is the min, bin size for 'Speech2Text' transformer model 
    print(f"k = {k}")
    
    # print(f"k = {k}")
    r2t = np.zeros(1)
    r2v = np.zeros(1)
    r2tt = np.zeros(1)
    pct = np.zeros(1)
    pcv = np.zeros(1)
    pctt = np.zeros(1)

    #downsamples if k>1 
    if k >1:
      feats = utils.down_sample(self.features[layer], k)
    else:
      feats = self.features[layer]

    y = self.simply_spikes(ch=channel, delay=delay)
    if k>1:
      y = utils.down_sample(y,k)
    m = int(feats.shape[0])
    n2 = int(m*0.9)
    x_test = feats[n2:, :]
    y_test = y[n2:]    
    
    for i in range(5):
        a = int(i*0.2*n2)
        b = int((i+1)*0.2*n2)
        
        x_val = feats[a:b, :] 
        y_val = y[a:b] 
        
        x_train = np.concatenate((feats[:a,:], feats[b:n2,:]), axis=0)
        y_train = np.concatenate((y[:a], y[b:n2]))
        # Linear Regression...!
        B = self.regression_param(x_train, y_train)
        y_hat_train = self.predict(x_train, B)
        y_hat_val = self.predict(x_val, B)
        y_hat_test = self.predict(x_test, B)
        
        pct += np.corrcoef(y_hat_train, y_train)[0,1]
        pcv += np.corrcoef(y_hat_val, y_val)[0,1]
        pctt += np.corrcoef(y_hat_test, y_test)[0,1]
        
    pct /= 5
    pcv /= 5
    pctt /= 5
    
    return pct, pcv,pctt



  def FE_r2_channel(self, layer, win, channel):
    k = int(win/40)    # 40 is the min, bin size for 'Speech2Text' transformer model 
    print(f"k = {k}")
    r2t = np.zeros(1)
    r2v = np.zeros(1)
    pct = np.zeros(1)
    pcv = np.zeros(1)
    #downsamples if k>1 
    if k >1:
      feats = self.down_sample_features(self.demean_features[layer], k)
    else:
      feats = self.demean_features[layer]

    m = int(feats.shape[0] *0.75)
    x_train = feats[0:m, :]
    x_test = feats[m:, :]
    
    # for i in range(self.dataset.num_channels):
    y = self.demean_spikes(ch=channel)
    if k>1:
      y = self.down_sample_spikes(y,k)
    y_train = y[0:m]
    y_test = y[m:]
    B = self.regression_param(x_train, y_train)

    r2t = self.regression_score(x_train, y_train, B)
    r2v = self.regression_score(x_test, y_test, B)
    pct = (np.corrcoef(self.predict(x_train, B), y_train)[0,1])**2
    pcv = (np.corrcoef(self.predict(x_test, B), y_test)[0,1])**2
    return r2t, r2v, pct, pcv

#   def signal_power(self, win, ch):
#     sents = [12,13,32,43,56,163,212,218,287,308]
#     sp = 0
#     for s in sents:
#         r = self.dataset.retrieve_spike_counts_for_all_trials(sent=s, w=win)[ch]
#         N = r.shape[0]
#         s = np.sum(r, axis=0)
#         n1 = np.var(s, axis=0)
#         n2 = 0
#         for i in range(r.shape[0]):
#             n2 += np.var(r[i])
#         sp += (n1 - n2)/(N*(N-1))
#     sp /= len(sents)
#     return sp 



  # def translate(self, aud, fs = 16000):
  #   if self.model_name == 'speech2text':
  #       inputs_features = self.processor(aud,padding=True, sampling_rate=fs, return_tensors="pt").input_features
  #   elif self.model_name == 'wav2vec':
  #       inputs_features = self.prepare_wav2vec_input(aud, fs)
  #   elif self.model_name == 'gru':
  #       inputs_features = self.prepare_GRU_input(aud)
  #   else:
  #       inputs_features = aud
  #   generated_ids = self.model_extractor(inputs_features)
