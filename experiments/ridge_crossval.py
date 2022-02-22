from auditory_cortex.Regression import transformer_regression
from auditory_cortex.ridge_regression import RidgeRegression 


# import sys
import json
from matplotlib.font_manager import json_dump
import yaml
from yaml import Loader
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
# from sklearn.metrics import r2_score

import time
# import csv

conf_path = '/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/conf/ridge_conf.yaml'

# conf_path = '/Users/akshita/Documents/Research/Makin/Auditory_Cortex/conf/ridge_conf.yaml'
with open(conf_path, "r") as f:
    manifest = yaml.load(f, Loader=Loader)

data_path = manifest['data_path']
subject = manifest['sub']
output_dir = manifest['output_dir']

# data_path = '/Users/akshita/Documents/Research/Makin/data'
# subject = '200206'
# output_dir = '/Users/akshita/Documents/Research/Makin/Auditory_Cortex'

reg = transformer_regression(data_path, subject)

test_list = np.arange(450,499).tolist()
train_val_list = np.arange(1,450)
# train_list = np.arange(1,450).tolist()
w = 80
sp = 1

num_layers = len(reg.layers)


# alphas = [0, 1, 10, 100, 1000, 10000, 100000]
# alphas = [1000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000]
alphas = np.linspace(1e6, 1e7, 10).tolist()

kf = KFold(n_splits=5, shuffle=True)


for alpha in alphas:
    print("alpha: ", alpha)
    tot_layer = {}
    start = time.time()
    for l in range(num_layers):
       
        r2t_tot = []
        r2v_tot = []
        r2tt_tot = []
        z_vals_test, n_vals_test = reg.get_layer_values_and_spikes(layer=l, win=w, sent_list=test_list)
        
        for train, val in kf.split(train_val_list):
            z_vals_train, n_vals_train = reg.get_layer_values_and_spikes(layer=l, win=80, sent_list=train_val_list[train])
            z_vals_val, n_vals_val = reg.get_layer_values_and_spikes(layer=l, win=80, sent_list=train_val_list[val])

            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(z_vals_train, n_vals_train)
            n_hat_train = ridge_model.predict(z_vals_train)
            n_hat_val = ridge_model.predict(z_vals_val)
            n_hat_test = ridge_model.predict(z_vals_test)

            r2t = []
            r2v = []
            r2tt = []
            
            for i in range(reg.dataset.num_channels):
                r2t.append(reg.cc_norm(n_hat_train[:,i], n_vals_train[:,i], sp=sp))
                r2v.append(reg.cc_norm(n_hat_val[:,i], n_vals_val[:,i], sp=sp))
                r2tt.append(reg.cc_norm(n_hat_test[:,i], n_vals_test[:,i], sp=sp))
            
            # break
            
        r2t_tot.append(r2t)
        r2v_tot.append(r2v)
        r2tt_tot.append(r2tt)

        r2t_tot = np.average(r2t_tot, axis = 0)
        r2v_tot = np.average(r2v_tot, axis = 0)
        r2tt_tot = np.average(r2tt_tot, axis = 0)

        tot_layer[l] = {'train': r2t_tot.tolist(), 'val': r2v_tot.tolist(), 'test': r2tt_tot.tolist()}
        # tot_layer[l] = {'train': r2t.tolist(), 'val': r2v.tolist(), 'test': r2tt.tolist()}

    
    # print("time: ",time.time()-start)
    # break
    with open(output_dir + "/" + subject + '_' + str(alpha), 'w') as f:
        json.dump(tot_layer, f)