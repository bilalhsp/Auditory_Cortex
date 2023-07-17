import os
import numpy as np
from scipy.io import wavfile
from auditory_cortex import config, opt_inputs_dir
import auditory_cortex.analysis.analysis as analysis
from auditory_cortex.optimal_input import OptimalInput


optimal_inputs_param = config['optimal_inputs_param']
model_name = optimal_inputs_param['model_name']
start_sent = optimal_inputs_param['starting_sent']
threshold = optimal_inputs_param['threshold']
force_redo = optimal_inputs_param['force_redo']
sessions = optimal_inputs_param['sessions']
layers = optimal_inputs_param['layers']
# channels = optimal_inputs_param['channels']


opt_obj = OptimalInput(model_name, load_features=True)
corr_obj = analysis.Correlations(model_name)

if sessions is None:
    sessions = corr_obj.get_significant_sessions(threshold=threshold)

for session in sessions:
    layer_channels_done = []
    # check if the directory already exists or not.
    sub_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles', str(int(session)))
    if not os.path.exists(sub_dir):
        print(f"Creating directory: {sub_dir}")
        os.makedirs(sub_dir)
    else:
        filenames = os.listdir(sub_dir)
        for filename in filenames:
            ind = filename.rfind('.wav')
            starting_sent = int(filename[ind-3:ind])
            if starting_sent == start_sent:    
                ind = filename.rfind('_corr_')
                layer_channels_done.append(filename[ind-5:ind])
    
    channels = corr_obj.get_good_channels(session, threshold=threshold)
    for layer in layers:
        print(f"For layer-{layer}")
        for ch in channels:
            print(f"\t ch-{ch}", end='\t')

            if f'{layer:02.0f}_{ch:02.0f}' not in layer_channels_done or force_redo:                
                inputs, losses, basic_loss, TVloss, grads = opt_obj.get_optimal_input(
                        session=session, layer=layer, ch=ch, starting_sent=start_sent,
                    )
                corr = corr_obj.get_corr_score(session, layer, ch)
                # saving the optimal
                optimal = np.array(inputs[-1].squeeze())
                filename = f"optimal_{model_name}_{session}_{layer:02.0f}_{ch:02.0f}_corr_{corr:.2f}_starting_{start_sent:03d}.wav"
                path = os.path.join(sub_dir, filename)
                wavfile.write(path, 16000, optimal.astype(np.float32))
                print(f"Saved optimal wavefile for, layer: {layer}, ch: {ch}...!")
            
            else:
                print(f"optimal wavefile already exists.")

