import time
import numpy as np
import cupy as cp
from auditory_cortex import utils

def cross_validted_regression(X,Y, iterations=1, num_lmbdas=10, train_val_size=0.8,
                              num_folds=5, use_gpu=True, verbose=False):    
    """Returns correlations and betas"""

    stim_ids = list(X.keys())
    num_train_val_stim = int(train_val_size*len(stim_ids))
    
    # num_X_features = X[stim_ids[0]].shape[-1]
    num_Y_features = Y[stim_ids[0]].shape[-1]
  
    # feature_dims = self.sampled_features[0].shape[1]
    # Beta = np.zeros((feature_dims, num_channels))
    lmbdas = np.logspace(start=-4, stop=-1, num=num_lmbdas)
    corr_coeff = []    
    Beta = [] 
    if verbose: 
        print(f"# of iterations requested: {iterations}, \n"+ \
                "# of lambda samples per iteration: {len(lmbdas)}")
    time_itr = 0
    time_lmbda = 0
    time_map = 0
    # time_fold = 0
    for n in range(iterations): 
        if verbose:
            print(f"Itr: {n+1}:")
        start_itr = time.time()
    
        np.random.shuffle(stim_ids)
        
        train_val_stim = stim_ids[:num_train_val_stim]
        test_stim = stim_ids[num_train_val_stim:]
        
        # lmbda_loss = module.zeros(((len(lmbdas), num_channels, self.num_layers)))
        start_lmbda = time.time()
        lmbda_loss = k_fold_cross_validation(
            X, Y, train_val_stim, lmbdas=lmbdas, num_folds=num_folds,
            use_gpu=use_gpu
            )
        end_lmbda = time.time()
        time_lmbda += end_lmbda-start_lmbda
        optimal_lmbdas = lmbdas[np.argmin(lmbda_loss, axis=0)]
        start_map = time.time()
        # Loading Mapping set...!
        # train_x = np.concatenate([X[s] for s in train_val_stimuli], axis=0)
        # train_y = np.concatenate([Y[s] for s in train_val_stimuli], axis=0)
        train_val_x = unroll_data(X, train_val_stim, use_gpu=use_gpu)
        train_val_y = unroll_data(Y, train_val_stim, use_gpu=use_gpu)
        test_x = unroll_data(X, test_stim, use_gpu=use_gpu)
        test_y = unroll_data(Y, test_stim, use_gpu=use_gpu)
        # Loading test set...!
        # test_x = np.concatenate([X[s] for s in test_stim], axis=0)
        # test_y = np.concatenate([Y[s] for s in test_stim], axis=0) 

        for ch in range(num_Y_features):
            Beta.append(utils.reg(train_val_x, train_val_y[:,ch], optimal_lmbdas[ch]))

        # coule be cp or np arrays..
        xp = cp.get_array_module(Beta[0])
        Beta = xp.array(Beta).transpose()

        # Beta = utils.reg(train_x, train_y, optimal_lmbda)
        test_pred = utils.predict(test_x, Beta)
        corr_coeff.append(utils.cc_norm(test_y,test_pred))


        end_map = time.time()
        end_itr = time.time()
        time_map += end_map - start_map
        time_itr += (end_itr - start_itr)
    if verbose:
        print(f"It takes (on avg.) {time_lmbda/(iterations):.2f} sec (all lmbdas). (time for {num_folds}-folds)")
        print(f"It takes (on avg.) {time_map/(iterations):.2f} sec/mapping.")
        print(f"It takes (on avg.) {time_itr/(iterations*60):.2f} minutes/iteration...!")
    corr_coeff = np.array(corr_coeff)

    return np.median(corr_coeff, axis=0), Beta
        

def k_fold_cross_validation(X, Y, train_val_stim_ids, lmbdas, num_folds=5,
                            use_gpu=True):

    size_of_chunk = int(len(train_val_stim_ids)/num_folds)
    num_Y_features = Y[train_val_stim_ids[0]].shape[-1]
    lmbda_loss = np.zeros((len(lmbdas), num_Y_features))
    train_val_stim_ids = np.array(train_val_stim_ids)
        
    for r in range(num_folds):
        # get the sent ids for train and validation folds...
        if r<(num_folds-1):
            val_set = train_val_stim_ids[r*size_of_chunk:(r+1)*size_of_chunk]
        else:
            val_set = train_val_stim_ids[r*size_of_chunk:]
        train_set = train_val_stim_ids[np.isin(train_val_stim_ids, val_set, invert=True)]

        # load features and spikes using the sent ids.
        # train_x = np.concatenate([X[s] for s in train_set], axis=0)
        # train_y = np.concatenate([Y[s] for s in train_set], axis=0)
        train_x = unroll_data(X, train_set, use_gpu=use_gpu)
        train_y = unroll_data(Y, train_set, use_gpu=use_gpu)
        val_x = unroll_data(X, val_set, use_gpu=use_gpu)
        val_y = unroll_data(Y, val_set, use_gpu=use_gpu)

        # val_x = np.concatenate([X[s] for s in val_set], axis=0)
        # val_y = np.concatenate([Y[s] for s in val_set], axis=0)

        # for the current fold, compute/save validation loss for each lambda.
        for i, lmbda in enumerate(lmbdas):
            Beta = utils.reg(train_x, train_y, lmbda)
            val_pred = utils.predict(val_x, Beta)
            loss = utils.mse_loss(val_y, val_pred)
            lmbda_loss[i] += loss

    lmbda_loss /= num_folds            
    return lmbda_loss


def unroll_data(data, stim_ids, use_gpu=True):
    unrolled_data = np.concatenate([data[s] for s in stim_ids], axis=0)
    if not use_gpu:
        unrolled_data = cp.asarray(unrolled_data)
    return unrolled_data

