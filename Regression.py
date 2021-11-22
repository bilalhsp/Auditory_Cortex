import numpy as np
import torch
import os
from scipy import linalg
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from Auditory_Cortex.Dataset import Neural_Data
from Auditory_Cortex.Feature_Extractors import Feature_Extractor_S2T

class transformer_regression():
  def __init__(self, dir, subject):
    self.dir = os.path.join(dir, subject)
    self.dataset = Neural_Data(dir, subject)
    self.layers = ["model.encoder.layers.0.fc2", "model.encoder.layers.1.fc2", "model.encoder.layers.2.fc2","model.encoder.layers.3.fc2",
                   "model.encoder.layers.4.fc2","model.encoder.layers.5.fc2","model.encoder.layers.6.fc2","model.encoder.layers.7.fc2",
                   "model.encoder.layers.8.fc2","model.encoder.layers.9.fc2"]
    self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
    self.model_extractor = Feature_Extractor_S2T(self.model, self.layers)
    print("Objects created, now loading Transformer layer features...!")
    self.features, self.demean_features = self.get_transformer_features()

  def simply_spikes(self, sent_s=1, sent_e=499, ch=0, w = 40, delay=0):
    spikes ={}
    for x,i in enumerate(range(sent_s,sent_e)):
      spikes[x] = torch.tensor(self.dataset.retrieve_spike_counts(sent=i, win=w, delay=delay ,early_spikes=False)[ch])
    spikes = torch.cat([spikes[i] for i in range(sent_e - sent_s)], dim = 0).numpy()
    return spikes

  def demean_spikes(self, sent_s=1, sent_e=499, ch=0, w = 40):
    spikes ={}
    spk_mean = {}
    for x,i in enumerate(range(sent_s,sent_e)):
      spikes[x] = torch.tensor(self.dataset.retrieve_spike_counts(sent=i, win=40 ,early_spikes=False)[ch])
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

  def translate(self, aud, fs = 16000):
    inputs_features = self.processor(aud,padding=True, sampling_rate=fs, return_tensors="pt").input_features
    generated_ids = self.model_extractor(inputs_features)

  # def simply_stack(self, features):
  #   features = torch.cat([features[i] for i in range(498)], dim=0)
  #   return features

  # def get_transformer_features(self):
  #   sent_s = 1
  #   sent_e = 499
  #   features = [{} for _ in range(len(self.layers))]

  #   feats = {}
  #   for x, i in enumerate(range(sent_s, sent_e)):
  #     self.translate(self.dataset.audio(i))
  #     for j, l in enumerate(self.layers):
  #       features[j][x] = self.model_extractor.features[l]
    
  #   for j, l in enumerate(self.layers):
  #     feats[j] = torch.cat([features[j][i] for i in range(sent_e-sent_s)], dim=0).numpy()
  #   return feats

  def get_transformer_features(self):
    sent_s = 1
    sent_e = 499
    features = [{} for _ in range(len(self.layers))]
    demean_features = [{} for _ in range(len(self.layers))]
    f_mean = {}    
    feats = {}
    demean_feats = {}
    for x, i in enumerate(range(sent_s, sent_e)):
      self.translate(self.dataset.audio(i))
      for j, l in enumerate(self.layers):
        features[j][x] = self.model_extractor.features[l]
        f_mean[x] = torch.mean(features[j][x], dim = 0)    
        demean_features[j][x]= features[j][x] - f_mean[x]
    for j, l in enumerate(self.layers):
      feats[j] = torch.cat([features[j][i] for i in range(sent_e-sent_s)], dim=0).numpy()
      demean_feats[j] = torch.cat([demean_features[j][i] for i in range(sent_e-sent_s)], dim=0).numpy()
    return feats, demean_feats

  # for i in range(498):
  #   f_mean[i] = torch.mean(features[i], dim = 0)
  #   features[i] = features[i] - f_mean[i]
  # features = torch.cat([features[i] for i in range(498)], dim=0)

  def down_sample_features(self, feats, k):
    out = np.zeros((int(np.ceil(feats.shape[0]/k)),feats.shape[1]))
    for i in range(out.shape[0]):
      #Just add the remaining samples at the end...!
      if (i == out.shape[0] -1):
        out[i] = feats[k*i:, :].sum(axis=0)
      else:  
        out[i] = feats[k*i:k*(i+1), :].sum(axis=0)
    return out

  def down_sample_spikes(self, spks, k):
    out = np.zeros(int(np.ceil(spks.shape[0]/k)))
    for i in range(out.shape[0]):
      #Just add the remaining samples at the end...!
      if (i == out.shape[0] -1):
        out[i] = spks[k*i:].sum(axis=0)
      else:  
        out[i] = spks[k*i:k*(i+1)].sum(axis=0)
    return out

  def compute_r2(self, layer, win):
    k = int(win/40)    # 40 is the min, bin size for 'Speech2Text' transformer model 
    print(f"k = {k}")
    r2t = np.zeros(self.dataset.num_channels)
    r2v = np.zeros(self.dataset.num_channels)
    pct = np.zeros(self.dataset.num_channels)
    pcv = np.zeros(self.dataset.num_channels)

    #downsamples if k>1 
    if k >1:
      feats = self.down_sample_features(self.features[layer], k)
    else:
      feats = self.features[layer]

    m = int(feats.shape[0] *0.75)
    x_train = feats[0:m, :]
    x_test = feats[m:, :]
    
    for i in range(self.dataset.num_channels):
      y = self.simply_spikes(ch=i)
      if k>1:
        y = self.down_sample_spikes(y,k)
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
    r2t = np.zeros(1)
    r2v = np.zeros(1)
    r2tt = np.zeros(1)
    pct = np.zeros(1)
    pcv = np.zeros(1)
    pctt = np.zeros(1)

    #downsamples if k>1 
    if k >1:
      feats = self.down_sample_features(self.features[layer], k)
    else:
      feats = self.features[layer]

    n1 = int(feats.shape[0] *0.75)
    n2 = int(feats.shape[0] *0.90)
    x_train = feats[0:n1, :]
    x_val = feats[n1:n2, :]
    x_test = feats[n2:, :]
    
    # for i in range(self.dataset.num_channels):
    y = self.simply_spikes(ch=channel, delay=delay)
    if k>1:
      y = self.down_sample_spikes(y,k)
    y_train = y[0:n1]
    y_val = y[n1:n2]    
    y_test = y[n2:]
    B = self.regression_param(x_train, y_train)

    r2t = self.regression_score(x_train, y_train, B)
    r2v = self.regression_score(x_val, y_val, B)
    r2tt = self.regression_score(x_test, y_test, B)
    pct = np.corrcoef(self.predict(x_train, B), y_train)
    pcv = np.corrcoef(self.predict(x_val, B), y_val)
    pctt = np.corrcoef(self.predict(x_test, B), y_test)
    pct = np.square(pct[0,1])
    pcv = np.square(pcv[0,1])
    pctt = np.square(pctt[0,1])
    
    return r2t, r2v,r2tt, pct, pcv,pctt
  
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

  def regression_param(self, X, y):
    B = linalg.lstsq(X, y)[0]
    return B
  def predict(self, X, B):
    return X@B
