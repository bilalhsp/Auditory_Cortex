import os
import pickle
from auditory_cortex import opt_inputs_dir



def write_model_parameters(
        model_name, session, coefficents):
    """Writes/updates model parameters at path 
    determined using the model_name.
    """
    # check if path exists, create if doesn't exist.
    dirpath = os.path.join(opt_inputs_dir, model_name)
    if not os.path.exists(dirpath):
        print("Does not exist")
        os.makedirs(dirpath)

    # loading existing betas or creating new (if not available already)
    filepath = os.path.join(dirpath, f"{model_name}_beta_bank.pkl")
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            beta_bank = pickle.load(f)
            print("Loading file...")
    else:
        print(f"Creating new beta bank")
        beta_bank = {}

    beta_bank[session] = coefficents

    with open(filepath, 'wb') as f:
        pickle.dump(beta_bank, f) 

    print(f"Parameters computed and saved for {model_name}, sess-{session}")
