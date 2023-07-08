from auditory_cortex.models import Regression
from auditory_cortex.ridge_regression import RidgeRegression 


import sys
import yaml
from yaml import Loader
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# import time
import json


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

reg = Regression(data_path, subject)

test_list = np.arange(450,499).tolist()
train_val_list = np.arange(1,450)
# train_list = np.arange(1,450).tolist()
w = 80
sp = 1

num_layers = len(reg.layers)


# alphas = [0, 1, 10, 100, 1000, 10000, 100000]

kf = KFold(n_splits=5, shuffle=True)

tot_layer = {}

for l in range(num_layers):
    r2t_tot = []
    r2v_tot = []
    r2tt_tot = []
    z_vals_test, n_vals_test = reg.get_layer_values_and_spikes(layer=l, win=w, sent_list=test_list)
    for train, val in kf.split(train_val_list):

        
        # test data
        
        # start = time.time()
        z_vals_train, n_vals_train = reg.get_layer_values_and_spikes(layer=l, win=80, sent_list=train_val_list[train])
        z_vals_val, n_vals_val = reg.get_layer_values_and_spikes(layer=l, win=80, sent_list=train_val_list[val])
        
        # k_val_train , def_w_train, offset_train, z_vals_train = reg.get_layer_values(layer=l, win=w, sent_list=train_val_list[train])
        # n_vals_train = reg.get_all_channels(def_w=def_w_train, offset=offset_train, k_val=k_val_train, sent_list=train_val_list[train])

        # k_val_val , def_w_val, offset_val, z_vals_val = reg.get_layer_values(layer=l,win=w, sent_list=train_val_list[val])
        # n_vals_val = reg.get_all_channels(def_w=def_w_val, offset=offset_val, k_val=k_val_val, sent_list=train_val_list[val])

        B = reg.regression_param(z_vals_train, n_vals_train)
        y_hat_train = reg.predict(z_vals_train, B)
        y_hat_val = reg.predict(z_vals_val, B)
        y_hat_test = reg.predict(z_vals_test, B)
        ### Ridge Regression
        # ridge_model = Ridge(alpha=alpha)
        # ridge_model.fit(x_train, y_train)
        # y_hat_train = ridge_model.predict(x_train)
        # y_hat_val = ridge_model.predict(x_val)
        # y_hat_test = ridge_model.predict(x_test)

        r2t = []
        r2v = []
        r2tt = []
        #Normalized correlation coefficient
        for i in range(reg.dataset.num_channels):
            r2t.append(reg.cc_norm(y_hat_train[:,i], n_vals_train[:,i], sp=sp))
            r2v.append(reg.cc_norm(y_hat_val[:,i], n_vals_val[:,i], sp=sp))
            r2tt.append(reg.cc_norm(y_hat_test[:,i], n_vals_test[:,i], sp=sp))

        r2t_tot.append(r2t)
        r2v_tot.append(r2v)
        r2tt_tot.append(r2tt)

    r2t_tot = np.average(r2t_tot, axis = 0)
    r2v_tot = np.average(r2v_tot, axis = 0)
    r2tt_tot = np.average(r2tt_tot, axis = 0)

    tot_layer[l] = {'train': r2t_tot.tolist(), 'val': r2v_tot.tolist(), 'test': r2tt_tot.tolist()}
    

    with open(output_dir + "/" + subject + '_lin_reg_allchannel', 'w') as f:
        json.dump(tot_layer, f)


        # r2t += reg.cc_norm(y_hat_train, n_vals_train, sp=sp)
        # r2v += reg.cc_norm(y_hat_val, n_vals_val, sp=sp)
        # r2tt += reg.cc_norm(y_hat_test, n_vals_test, sp=sp)
        
        
    # with open(output_dir + "/" + subject + '_linear_reg' +".csv" ,'a') as f1:
    #         writer=csv.writer(f1)
    #         row = [subject, l, r2t/5, r2v/5, r2tt/5 ]
    #         writer.writerow(row)
    #         f1.close()

    # with open(output_dir + "/" + subject + '_' + str(int(alpha)) + '_all_layers' +".csv" ,'a') as f2:
    #     writer=csv.writer(f1)
    #     row = [subject, l, alpha, r2t/5, r2v/5, r2tt/5]
    #     writer.writerow(row)
    #     f1.close()


# from auditory_cortex.Regression import Regression

# import json 
# import sys
# import csv

# print(sys.argv)
# sub = sys.argv[1]
# # w = sys.argv[2]
# alpha = float(sys.argv[2])
# # print("arg vals",sub, int(w))
# w = 80

# # print(alpha, alpha/10, alpha/10.0)
# reg = Regression('/depot/jgmakin/data/auditory_cortex/josh_data/data',sub)

# # channels = np.arange(0,reg.dataset.num_channels).tolist()5
# # alphas = [0.1, 0.2, 0.5, 0.8, ]

# num_layers = len(reg.layers)
# corr_values = {}
# for ch in range(reg.dataset.num_channels):
#     R2t =[]  
#     R2v =[] 
#     R2tt =[]
#     for l in range(num_layers):
#         r2t, r2v,r2tt = reg.get_cc_norm(l,w,channel=ch, delay=0, alpha=alpha/10.0)
#         R2t.append(r2t.item())
#         R2v.append(r2v.item())
#         R2tt.append(r2tt.item())
#         # PCt.append(pct)
#         # PCv.append(pcv)
#         # PCtt.append(pctt)
#     corr_values[ch] =  {"train": R2t, "val": R2v, "test": R2tt}
# # fig, ax = plt.subplots(1,2, figsize=(14,6), sharey=True)
#     with open("/scratch/gilbreth/akamsali/Research/Makin/outputs/neuron_corr/ridge_regression_alphas/"+ sub + "_" + str(w) + '_'+str(alpha) +'.csv' ,'a') as f1:
#         writer=csv.writer(f1)
#         row = R2t + R2v + R2tt
#         writer.writerow(row)
#         f1.close()

# with open("/scratch/gilbreth/akamsali/Research/Makin/outputs/neuron_corr/ridge_regression_alphas/" + sub + "_" + str(w)+ '_'+str(alpha), 'w') as f:
#     json.dump(corr_values, f)
