from auditory_cortex.Regression import transformer_regression

import json 
import sys
import csv

print(sys.argv)
sub = sys.argv[1]
w = sys.argv[2]

# print("arg vals",sub, int(w))

reg = transformer_regression('/depot/jgmakin/data/auditory_cortex/josh_data/data',sub)

# channels = np.arange(0,reg.dataset.num_channels).tolist()
num_layers = len(reg.layers)
corr_values = {}
for ch in range(reg.dataset.num_channels):
    R2t =[]  
    R2v =[] 
    R2tt =[]
    for l in range(num_layers):
        r2t, r2v,r2tt = reg.get_cc_norm(l,int(w),channel=ch, delay=0)
        R2t.append(r2t.item())
        R2v.append(r2v.item())
        R2tt.append(r2tt.item())
        # PCt.append(pct)
        # PCv.append(pcv)
        # PCtt.append(pctt)
    corr_values[ch] =  {"train": R2t, "val": R2v, "test": R2tt}
# fig, ax = plt.subplots(1,2, figsize=(14,6), sharey=True)
    with open("/scratch/gilbreth/akamsali/Research/Makin/outputs/neuron_corr/with_sp/"+ sub + "_" + str(w) +'.csv' ,'a') as f1:
        writer=csv.writer(f1)
        row = R2t + R2v + R2tt
        writer.writerow(row)
        f1.close()

with open("/scratch/gilbreth/akamsali/Research/Makin/outputs/neuron_corr/with_sp/" + sub + "_" + str(w), 'w') as f:
    json.dump(corr_values, f)
