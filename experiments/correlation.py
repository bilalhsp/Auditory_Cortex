from auditory_cortex.Regression import transformer_regression

import json 


sub = "200213"
w = 100


reg = transformer_regression('/depot/jgmakin/data/auditory_cortex/josh_data/data',sub)


# channels = np.arange(0,reg.dataset.num_channels).tolist()
num_layers = len(reg.layers)
corr_values = {}
for ch in range(4):
    R2t =[]  
    R2v =[] 
    R2tt =[]
    for l in range(num_layers):
        r2t, r2v,r2tt = reg.get_cc_norm(l,w,channel=ch, delay=0)
        R2t.append(r2t.item())
        R2v.append(r2v.item())
        R2tt.append(r2tt.item())
        # PCt.append(pct)
        # PCv.append(pcv)
        # PCtt.append(pctt)
    corr_values[ch] =  {"train": R2t, "val": R2v, "test": R2tt}
# fig, ax = plt.subplots(1,2, figsize=(14,6), sharey=True)

with open("/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/"+sub + "_" + str(w), 'a') as f:
    json.dump(corr_values, f)