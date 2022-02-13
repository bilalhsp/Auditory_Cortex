from re import sub
from auditory_cortex.Regression import transformer_regression
from auditory_cortex.ridge_regression import RidgeRegression 


import sys
import yaml
from yaml import Loader
import numpy as np
from sklearn.model_selection import KFold
import time
import csv


conf_path = '/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/conf/ridge_conf.yaml'
with open(conf_path, "r") as f:
    manifest = yaml.load(f, Loader=Loader)

data_path = manifest['data_path']
subject = manifest['sub']
output_dir = manifest['output_dir']


reg = transformer_regression(data_path, subject)

test_list = np.arange(450,499).tolist()
train_val_list = np.arange(1,450)
w = 80
sp = 1
num_layers = len(reg.layers)

alphas = [0, 1, 10, 100, 1000, 10000, 100000]

for l in range(num_layers):
    k_val_test , def_w_test, offset_test, z_vals_test = reg.get_layer_values(l, win=w, sent_list=test_list)
    n_vals_test = reg.get_all_channels(def_w=def_w_test, offset=offset_test, k_val=k_val_test, sent_list=test_list)
    # test data
    # r2t = 0
    # r2v = 0
    # r2tt = 0
    kf = KFold(n_splits=5, shuffle=True)
    for train, val in kf.split(train_val_list):
        start = time.time()
        
        k_val_train , def_w_train, offset_train, z_vals_train = reg.get_layer_values(layer=l, win=w, sent_list=train_val_list[train])
        n_vals_train = reg.get_all_channels(def_w=def_w_train, offset=offset_train, k_val=k_val_train, sent_list=train_val_list[train])

        k_val_val , def_w_val, offset_val, z_vals_val = reg.get_layer_values(layer=l,win=w, sent_list=train_val_list[val])
        n_vals_val = reg.get_all_channels(def_w=def_w_val, offset=offset_val, k_val=k_val_val, sent_list=train_val_list[val])

        for alpha in alphas:
            ridge_model = RidgeRegression(alpha=alpha)
            ridge_model.fit(z_vals_train, n_vals_train)
            n_hat_train = ridge_model.predict(z_vals_train)
            n_hat_val = ridge_model.predict(z_vals_val)

            n_hat_test = ridge_model.predict(z_vals_test)
            r2t = reg.cc_norm(n_hat_train, n_vals_train, sp=sp)
            r2v = reg.cc_norm(n_hat_val, n_vals_val, sp=sp)
            r2tt = reg.cc_norm(n_hat_test, n_vals_test, sp=sp)
            
            with open(output_dir + "/" + subject + '_over_alphas' +".csv" ,'a') as f1:
                    writer=csv.writer(f1)
                    row = [subject, l, alpha, r2t, r2v, r2tt]
                    writer.writerow(row)
                    f1.close()
    
    # with open(output_dir + "/" + subject + '_' + str(int(alpha)) + '_all_layers' +".csv" ,'a') as f2:
    #     writer=csv.writer(f1)
    #     row = [subject, l, alpha, r2t/5, r2v/5, r2tt/5]
    #     writer.writerow(row)
    #     f1.close()
        

