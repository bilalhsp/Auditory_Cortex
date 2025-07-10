# adjsut the basic logging lovel of notebook
import logging
logging.basicConfig(level=logging.WARNING)

import os
import pickle
from auditory_cortex import optimal_input, opt_inputs_dir
from auditory_cortex.analyses.deprecated import regression_correlations

# model_name = 'deepspeech2'
# model_name = 'speech2text'
model_name = 'wav2vec2'
# model_name = 'wav2letter_modified'
threshold = 0.068

print(f"Creating objects for {model_name}")
results_id = 'opt_neural_delay'
corr_file = model_name + '_' + results_id
corr_obj = regression_correlations.Correlations(model_name=corr_file)
opt_inp = optimal_input.OptimalInput(model_name=model_name, load_features=True)

   
dirpath = os.path.join(opt_inputs_dir, model_name)
if not os.path.exists(dirpath):
    print("Does not exist")
    os.makedirs(dirpath)


filepath = os.path.join(dirpath, f"{model_name}_beta_bank.pkl")
if os.path.exists(filepath):
    with open(filepath, 'rb') as f:
        beta_bank = pickle.load(f)
        print("Loading file...")
else:
    print(f"Creating new beta bank")
    beta_bank = {}

sessions = corr_obj.get_significant_sessions(threshold=threshold)
sessions = sessions

print(f"Starting for loop")
for session in sessions:
    session = str(int(session))

    if session not in beta_bank.keys():
        beta = opt_inp.get_betas(session)
        


        beta_bank[session] = beta

        with open(filepath, 'wb') as f:
            pickle.dump(beta_bank, f) 
            print(f"Beta computed and saved for sess-{session}")
