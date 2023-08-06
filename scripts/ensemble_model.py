# adjsut the basic logging lovel of notebook
import os
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from auditory_cortex import utils, config, saved_corr_dir
from auditory_cortex.models import Regression
import pandas as pd




data_dir = config['neural_data_dir']
bad_sessions = config['bad_sessions']

## read the sessions available in data_dir
sessions = np.array(os.listdir(data_dir))
sessions = np.delete(sessions, np.where(sessions == "out_sentence_details_timit_all_loudness.mat"))
for s in bad_sessions:
    sessions = np.delete(sessions, np.where(sessions == s))


models = [
    {
        'model_name': 'wave2letter_modified',
        'layer': 6
    },
    {
        'model_name': 'speech2text',
        'layer': 3
    },
    {
        'model_name': 'wave2vec',
        'layer': 8
    },
    {
        'model_name': 'wave2vec2',
        'layer': 7
    },
    {
        'model_name': 'deepspeech2',
        'layer': 2
    },
]


# filepath = os.path.join(os.path.dirname(__file__), 'ensemble_correlations.csv')
filepath = os.path.join(saved_corr_dir, 'ensemble_corr_results.csv')

if os.path.exists(filepath):
    df = pd.read_csv(filepath)
else:
    column_names = ['session']
    column_names.extend([model['model_name'] for model in models])
    column_names.append('ensemble_model')
    
    df = pd.DataFrame(columns=column_names)

sessions_done = df['session'].unique()
sessions = sessions[np.isin(sessions, sessions_done, invert=True)]



reg_objs = []
# session = 200206

# create models..
for i, model in enumerate(models):
    model_name = model['model_name']
    reg_objs.append(Regression(model_name, load_features=True))

    

# # creating dataframe to store results
# column_names = ['session']
# column_names.extend([model['model_name'] for model in models])
# column_names.append('ensemble_model')

# df = pd.DataFrame(columns=column_names)

for session in sessions:
    ensemble_prediction = []
    correlations = []
    predictions = []


    # compute Beta
    for i, model in enumerate(models):
        if i==0:
            test_set = None
        corr_coeff, B, loss, test_set = reg_objs[i].cross_validated_regression(
            session=session, num_lmbdas=10, iterations=1, test_sents=test_set
            )

    spikes = reg_objs[0].get_neural_spikes(session, sents=test_set, numpy=True)

    for i, model in enumerate(models):
        predictions.append(reg_objs[i].neural_prediction(session, sent=test_set))
        
    for i, model in enumerate(models):
        layer = model['layer']
        layer_idx = reg_objs[i].get_layer_index(layer)
        ensemble_prediction.append(predictions[i][:,:,layer_idx])
        correlations.append(np.median(utils.cc_norm(spikes, predictions[i][:, :, layer_idx])))

    ensemble_prediction = sum(ensemble_prediction)/len(models)
    ensemble_corr = np.median(utils.cc_norm(spikes, ensemble_prediction))

    # adding data for this session
    row = [session]
    row.extend(correlations)
    row.append(ensemble_corr)

    df.at[len(df.index)] =  row

    df.to_csv(filepath, index=False)
    print(f"Done for session: {session}")




