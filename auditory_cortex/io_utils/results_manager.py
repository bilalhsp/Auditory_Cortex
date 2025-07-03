"""This module is to handle saving results in a structured way.
This also includes testing if results are available for all
layers of the model and merging results from different runs.

Class ResultsManager:
    - saves results in a structured way
    - checks if results are available for all layers
    - merges results of all layers

"""
import os
import numpy as np
import pandas as pd

from auditory_cortex import utils, results_dir, aux_dir, saved_corr_dir

import logging
logger = logging.getLogger(__name__)


class ResultsManager:

    @staticmethod
    def get_run_id(
        dataset_name, bin_width, identifier, mVocs=False, shuffled=False, lag=300, 
        **kwargs
    ):
        """Returns the identifier for the results file. run_trf2.py script uses
        the selection parameters to create the identifier for the results file.
        This helps keep the results organized and easy to find.
        Args:

            dataset_name: str = name of the dataset, e.g. 'ucsf', 'ucdavis'
            bin_width: int = bin width in ms, e.g. 50, 100
            identifier: str = identifier for the results file, e.g. 'plos_view'
            mVocs: bool = True if mVocs dataset, False if TIMIT dataset
            shuffled: bool = True if shuffled, False if not
            lags: int = number of lags to use for the TRF, default=300

            kwargs: 
                - bootstrap: bool = True if running bootstrap, False if not. Default=False
                - test_bootstrap: bool: True if running bootstrap for test set, False if
                    running for training set. Default=False

        Returns:    
            identifier: str = identifier for the results file
        """
        bootstrap = kwargs.get('bootstrap', False)
        test_bootstrap = kwargs.get('test_bootstrap', False)

        full_identifier = dataset_name+'_'
        if shuffled:
            full_identifier += 'reset_'
        if mVocs:
            full_identifier += 'mVocs_'
        else:
            full_identifier += 'timit_'
        full_identifier += f'trf_lags{lag}_bw{bin_width}'

        if bootstrap:
            if test_bootstrap:
                full_identifier += '_bootstrap_test'
            else:
                full_identifier += '_bootstrap'
        if identifier != '':
            full_identifier += f'_{identifier}'

        logger.info(f"Results identifier: {full_identifier}")
        return full_identifier
    

    @staticmethod
    def check_results(model_name, identifier, num_sessions, verbose=False):
        """Display the number of sessions done for all bin widths, for 
        the given model and identifier.
        Returns a list of bin widths for which the sessions are not done,
        or None if all done.
        
        Args:
            model_name: str = name of the model
            identifier: str = identifier for the results file
            num_sessions: int = number of sessions to be done
            verbose: bool = True if verbose, False if not
        """
        return_list = []
        if verbose:
            logger.info(f"For '{model_name}', '{identifier}'")
        try:
            filename = f'{model_name}_{identifier}_corr_results.csv'
            corr_file_path = os.path.join(saved_corr_dir, filename)
            dataframe = pd.read_csv(corr_file_path)
        except FileNotFoundError:
            logger.warning(f"File not found: {corr_file_path}")
            return_list.append(model_name+'_'+identifier)
            return return_list

        bin_widths = np.sort(dataframe['bin_width'].unique())
        for bin_width in bin_widths:
            data = dataframe[dataframe['bin_width']==float(bin_width)]
            if verbose:
                logger.info(f"For bin_width: {bin_width:03} ms, sessions done: {len(data['session'].unique())}")
            if len(data['session'].unique()) != num_sessions:
                return_list.append(model_name+'_'+identifier+f'_bw{bin_width}')
            else:
                return_list.append(None)
        return return_list

    @staticmethod
    def check_results_across_all_layers(model_names, identifier, num_sessions, verbose=False):
        """Checks saved results for all the layers of the given model. Since my script saves a separate file 
        for every layer with layer number augmented to the identifier, this function checks all the layers
        for the given model and identifier. It returns a list of identifiers for which the sessions are not done.
        Runs the 'check_results' for all layers with 'identifier+'_l'+layer_id '.
        Args:
            model_names: list = list of model names
            identifier: str = identifier for the results file
            num_sessions: int = number of sessions to be done
            verbose: bool = True if verbose, False if not
        """
        models_not_done = []
        for model_name in model_names:
            model_config = utils.load_dnn_config(model_name=model_name)
            num_layers = len(model_config['layers'])
            for i in range(num_layers):
                not_done = ResultsManager.check_results(
                    model_name, identifier+f'_l{i}', num_sessions=num_sessions, verbose=verbose
                    )
                models_not_done.extend(not_done)
        # remove None entries..
        while None in models_not_done:
            models_not_done.remove(None)
        
        if len(models_not_done) ==0:
            logger.info(f"All models done..for {identifier}")
        else:
            logger.info(f"Models with incomplete resutls:")
            for iden in models_not_done:
                logger.info(iden)

    @staticmethod            
    def check_results_across_identifiers(model_names, identifiers_list, num_sessions, verbose=False):
        """Checks saved results for all the identifiers. .
        Args:
            model_names: list = list of model names
            identifiers_list: str = identifier for the results file
            num_sessions: int = number of sessions to be done
            verbose: bool = True if verbose, False if not
        """
        models_not_done = []
        for model_name in model_names:
            for iden in identifiers_list:
                not_done = ResultsManager.check_results(
                    model_name, iden, num_sessions=num_sessions, verbose=verbose
                    )
                models_not_done.extend(not_done)
        # remove None entries..
        while None in models_not_done:
            models_not_done.remove(None)
        
        if len(models_not_done) ==0:
            logger.info(f"All models done...for given identifiers")
        else:
            logger.info(f"Models with incomplete resutls:")
            for iden in models_not_done:
                logger.info(iden)

    @staticmethod
    def merge_correlation_results(
        model_name, identifiers_list, output_id=0, output_identifier=None
        ):
        """
        Args:

            model_name: Name of the pre-trained network
            file_identifiers: List of filename identifiers 
            idx:    id of the file identifier to use for saving the merged results
            output_identifier: if output_identifier is given, it takes preference..
        """
        logger.info("Combining results...")
        corr_dfs = []
        for iden in identifiers_list:
            filename = f"{model_name}_{iden}_corr_results.csv"
            file_path = os.path.join(saved_corr_dir, filename)
            corr_dfs.append(pd.read_csv(file_path))

        # save the merged results at the very first filename...
        if output_identifier is None:
            output_identifier = identifiers_list[output_id]    
        filename = f"{model_name}_{output_identifier}_corr_results.csv"
        file_path = os.path.join(saved_corr_dir, filename)

        data = pd.concat(corr_dfs)
        data.to_csv(file_path, index=False)
        logger.info(f"Output saved at: \n {file_path}")

        # once all the files have been merged, remove the files..
        for identifier in identifiers_list:
            if identifier != output_identifier:
                filename = f"{model_name}_{identifier}_corr_results.csv"
                file_path = os.path.join(saved_corr_dir, filename)
                # remove the file
                os.remove(file_path)

    @staticmethod
    def combine_results_for_all_models(model_names, identifier):
        """Combines results for the list of models provided"""
        for model_name in model_names:
            model_config = utils.load_dnn_config(model_name=model_name)
            num_layers = len(model_config['layers'])
            ids = [identifier+f'_l{i}' for i in range(num_layers)]
            ResultsManager.merge_correlation_results(
                model_name, ids, 0, 
                output_identifier=identifier
            )

