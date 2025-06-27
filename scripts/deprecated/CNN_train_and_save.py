import os
import time
import pickle
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
    
    num_layers = trial.suggest_int("num_layers", 1, 7)

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


def train_and_save(trial):

    batch_size = trial['batch_size']
    learning_rate = trial['learning_rate']
    weight_decay = trial['weight_decay']

    # get dataloaders...
    train_dataloader, test_dataloader, feat_channels, spike_channels = get_dataloaders(
        batch_size, pretrained=PRETRAINED, pretrained_features=PRETRAINED_FEATURES,
        neural_area=AREA
    )

    num_layers = trial['num_layers']

    layers = []
    num_units = []
    kernel_sizes = []
    for i in range(num_layers):
        num_units.append(trial[f"num_units_l{i}"])
        kernel_sizes.append(trial[f"kernel_size_l{i}"])

    model = Network(
        num_layers, num_units, kernel_sizes, feat_channels, spike_channels
        ).to(device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss_history = []
    val_loss_history = []
    for epoch in range(EPOCHS):
        # training_loss = 0
        print(f"{epoch+1}/{EPOCHS}:")

        training_loss = training_epoch(
            model, optimizer, train_dataloader, DEVICE)
        train_loss_history.append(training_loss)
        # Evaluate performace...
        test_score = evaluation_epoch(
                model, test_dataloader, DEVICE
            )
        
        val_loss_history.append(test_score)

        # trial.report(test_score, epoch)
        # if trial.should_prune:
            # assert False, "should_prune() should always return False with this pruner."
            
            #  raise optuna.exceptions.TrialPruned()
    losses = {
        'training': train_loss_history,
        'validation': val_loss_history 
    }

    if PRETRAINED:
        name_identifier = f'{MODEL_NAME}_layer_{LAYER_ID}_area_{AREA}'
    else:
        name_identifier = f'CNN_area_{AREA}'

    network_checkpoint = os.path.join(
        cache_dir, 'optuna', f'{name_identifier}.pth'
    )
    losses_filename = os.path.join(
        cache_dir, 'optuna', f'{name_identifier}_losses.pkl'
    )
    print(f"Writing results...")
    with open(losses_filename, 'wb') as fout:
        pickle.dump(losses, fout)

    save_checkpoint(model, optimizer, EPOCHS, network_checkpoint)
    return test_score


def save_checkpoint(model, optimizer, epoch, PATH):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }
    print(f"Checkpoint saved at {PATH}")
    torch.save(checkpoint, PATH)


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
        '-a','--neural_area', dest='neural_area', action='store', default='all',
        choices=['core', 'belt', 'all'],
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

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    train_and_save(trial.params)



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
