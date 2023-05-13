import os
import time
import yaml

#
import numpy as np
import matplotlib.pyplot as plt

# local
import auditory_cortex.optimal_input as op_inp


START = time.time()
reg_conf = '/home/ahmedb/projects/Wav2Letter/Auditory_Cortex/conf/regression_w2l.yaml'
with open(reg_conf, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



data_dir = config['data_dir']
bad_sessions = config['bad_sessions']
results_dir = config['results_dir']

optimal_inputs_param = config['optimal_inputs_param']
model = optimal_inputs_param['model']
sent = optimal_inputs_param['starting_sent']
force_redo = optimal_inputs_param['force_redo']
layers = optimal_inputs_param['layers']
channels = optimal_inputs_param['channels']





# creating instance of optimal_input object..
obj = op_inp.optimal_input(model, load_features=True)

if layers is None:
    layers = np.arange(obj.num_layers)


# delays = config['delays']
# bin_widths = config['bin_widths']
# pretrained = config['pretrained']
# k_folds_validation = config['k_folds_validation']
# iterations = config['iterations']
# use_cpu = config['use_cpu']
# dataset_sizes = config['dataset_sizes']
# dataset_sizes = np.arange(dataset_sizes[0], dataset_sizes[1], dataset_sizes[2])




## read the sessions available in data_dir, and remove bad channels
sessions = np.array(os.listdir(data_dir))
sessions = np.delete(sessions, np.where(sessions == "out_sentence_details_timit_all_loudness.mat"))
for s in bad_sessions:
    sessions = np.delete(sessions, np.where(sessions == s))



# sent = 00
# force_redo = False

# model = 'wave2letter_modified'
# layer = 6
# ch = 32
# sessions = ['180420']
# sessions = sessions[2:5]
# sessions = sessions[5:8]
sessions = ['200206']

for session in sessions:

    session_done_already = True
    # make sure sub-directories are created...
    path = os.path.join(results_dir, 'optimal_inputs', model, session)
    if not os.path.exists(path):
        print("Creating directory: \n {path} \n")
        os.makedirs(path)
        print(f"Done...!")

    # check which layer-channel pairs(for this starting sent) are already done... 
    layer_channels_done = []
    filenames = os.listdir(path)
    for filename in filenames:
        ind = filename.rfind('.jpg')
        starting_sent = int(filename[ind-3:ind])
        if starting_sent == sent:    
            ind = filename.rfind('_starting_')
            layer_channels_done.append(filename[ind-5:ind])

    if channels is None:
        obj.load_dataset(session)
        channels = np.arange(obj.get_num_channels(session))

    for layer in layers:
        for ch in channels:
            if f'{layer:02d}_{ch:02d}' not in layer_channels_done or force_redo:
                inputs, *output = obj.optimize(
                        session=session, layer=layer, ch=ch, starting_sent=sent,
                        )
                fig, ax = plt.subplots()
                op_inp.plot_spect(inputs[-1], ax)
                ax.set_title(f"{model}_{session}_{layer}_{ch}_starting_{sent}")

                fig_name = f"opt_input_{model}_{session}_{layer:02d}_{ch:02d}_starting_{sent:03d}.jpg"
                plt.savefig(os.path.join(path, fig_name))
                # setting 'session_done_already', to False, as optimization for this session was needed...
                print(f"Done with layer: {layer}, channel: {ch}")
                session_done_already = False
    if session_done_already:
        print(f"Session: {session} was already Done.")

end_time = time.time()
print(f"It took {end_time - START} seconds to run.")