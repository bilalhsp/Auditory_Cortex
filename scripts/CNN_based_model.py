import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import argparse

from auditory_cortex.spikes_dataset import SpikesData, Network, training_epoch, evaluation_epoch, get_dataloaders
from auditory_cortex import cache_dir
from auditory_cortex.deprecated.dataloader import DataLoader

SPIKE_CHANNELS = 217
SPECT_CHANNELS = 128
EPOCHS = 25


# BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_network(trial, in_channels, spike_channels):
    
    num_layers = trial.suggest_int("num_layers", 1, 15)

    layers = []
    num_units = []
    kernel_sizes = []
    for i in range(num_layers):
        num_units.append(trial.suggest_int(f"num_units_l{i}", 32, 512, log=True))
        kernel_sizes.append(trial.suggest_int(f"kernel_size_l{i}", 3, 9, step=2))

    model = Network(
        num_layers, num_units, kernel_sizes, in_channels, spike_channels
        )
    return model


def objective(trial):

    # pick values of hyper-parameters...    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int(f"batch_size", 4, 128, log=True)

    # get dataloaders...
    train_dataloader, test_dataloader, feat_channels, spike_channels = get_dataloaders(
        batch_size, pretrained=PRETRAINED, pretrained_features=PRETRAINED_FEATURES,
        neural_area=AREA
    )
    # create network...
    model = create_network(
        trial, in_channels=feat_channels, spike_channels=spike_channels
        ).to(device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(EPOCHS):
        # training_loss = 0
        print(f"{epoch+1}/{EPOCHS}:")

        training_loss = training_epoch(
            model, optimizer, train_dataloader, DEVICE)

        # Evaluate performace...
        test_score = evaluation_epoch(
                model, test_dataloader, DEVICE
            )

        trial.report(test_score, epoch)
        # if trial.should_prune:
            # assert False, "should_prune() should always return False with this pruner."
            
            #  raise optuna.exceptions.TrialPruned()
    return test_score


# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save the normalizer '+
            'for the sessions of neural data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # add arguments to read from command line
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',#  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec2', 'speech2text', 'deepspeech2',
                'whisper_tiny', 'whisper_small', 'whisper_base', 'whisper_medium'],
        default='wav2vec2', 
        help='model to be used for RSA analysis.'
    )
    parser.add_argument(
        '-a','--neural_area', dest='neural_area', action='store', 
        choices=['core', 'belt', 'all'],
        default='all',
        help="Choose bin_width for normalizers."
    )

    parser.add_argument(
        '-p','--pretrained', dest='pretrained', action='store_true', default=False,
        help="Choose bin_width for normalizers."
    )
    parser.add_argument(
        '-l','--layer', dest='layer_ID', action='store', type=int, default=7,
        help="layer ID of Neural areas."
    )
    return parser



def run_study(args):
    # pretrained = True
    # model_name = 'wav2vec2'
    # layer_ID = 7
    global PRETRAINED
    global MODEL_NAME
    global LAYER_ID
    global AREA
    global PRETRAINED_FEATURES

    PRETRAINED = args.pretrained
    MODEL_NAME = args.model_name
    LAYER_ID = args.layer_ID
    AREA = args.neural_area

    if PRETRAINED:
        dataloader = DataLoader()
        PRETRAINED_FEATURES = dataloader.get_DNN_layer_features(
                model_name=MODEL_NAME, layer_ID=LAYER_ID
            )
    else:
        PRETRAINED_FEATURES = None

    study_name = f"CNN_based_model_{AREA}"
    if PRETRAINED:
        study_name = f"{MODEL_NAME}_l{LAYER_ID}_based_model_{AREA}"

    optuna_cache_dir = os.path.join(cache_dir, 'optuna')
    if not os.path.exists(optuna_cache_dir):
        os.makedirs(optuna_cache_dir)
        print(f"Optuna cache dir created.")

    storage_name = f"sqlite:///{os.path.join(optuna_cache_dir, study_name)}.db"

    print(f"Creating optuna study...")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        # sampler=optuna.TPESampler(n_startup_trials=25, multivariate=True, seed=123),
        # pruner=optuna.pruners.NopPruner(),
        # pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource='auto',
        #         reduction_factor=4, min_early_stopping_rate=4
        #     ),
        # pruner=optuna.pruners.MedianPruner(
        #     n_startup_trials=5, n_warmup_steps=10, interval_steps=2,
        #     n_min_trials = 5
        # )  
    )

    study.optimize(objective, n_trials=25)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



if __name__ == "__main__":

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    run_study(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")
