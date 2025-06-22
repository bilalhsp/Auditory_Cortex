
import os
import pickle
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage

from auditory_cortex import opt_inputs_dir
from auditory_cortex import utils
from auditory_cortex.analyses.deprecated.regression_correlations import Correlations
from auditory_cortex.neural_data.deprecated.neural_meta_data import NeuralMetaData
from auditory_cortex.neural_data.deprecated.recording_config import RecordingConfig

class BetaAnalyzer:
    """Provides tools to analyze coefficients of linear 
    regression, and explore the topography of auditory cortex 
    using confusion matrix. 
    """
    def __init__(self, model_name=None) -> None:
        """
        Args:
            model_name (str): model name to be analyzed """

        self.model_name = model_name
        if model_name == 'strf_model':
            model_name = 'STRF_freqs128'
        else:
            model_name = model_name+'_'+'opt_neural_delay'
        self.corr_obj = Correlations(model_name=model_name)
        self.metadata = NeuralMetaData(RecordingConfig)
        self.beta_bank = self._read_betas()
        # first layer in beta_banks, this will be used to adjust layer index
        self.beta_first_layer = {
            'wav2letter_modified': 0,
            'speech2text': 2,
            'deepspeech2': 2,
            'wav2vec2': 7
        }

    def _read_betas(self):
        """Retrives regression weights, corresponding to the model,
        from the saved results directory."""
        # read betas...
        dirpath = os.path.join(opt_inputs_dir, self.model_name)
        filepath = os.path.join(dirpath, f"{self.model_name}_beta_bank.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results not saved, check and recompute...!")        
        else:
            with open(filepath, 'rb') as f:
                print("Loading file...")
                beta_bank = pickle.load(f)
        return beta_bank
    
    def _get_adjusted_layer_id(self, layer):
        """"Returns correct layer index for reading beta bank, adjusts
        the index as per stored results."""
        first_layer = self.beta_first_layer[self.model_name]
        adjusted_layer_id = layer - first_layer
        if adjusted_layer_id < 0:
            raise NotImplemented(f"regression weights (beta) not saved for layer-{layer}, try layer > {first_layer}. ")
        return adjusted_layer_id
        
        
    def get_sig_betas(self, ordered_sessions_list, layer=6, threshold = 0.068):
        """Returns list of betas for significant session/channels only.
        Args:
            ordered_sessions_list: list of recoring sessions in some 
                pre-determined (or even random) order.
            layer (int): ID of the layer to be analyzed.
        """
        ordered_sessions = np.array(ordered_sessions_list)
        if self.model_name != 'strf_model':
            layer = self._get_adjusted_layer_id(layer)
        # get only significant sessions...
        sig_sessions = self.corr_obj.get_significant_sessions(threshold=threshold)
        sig_ordered_sessions = ordered_sessions[np.isin(ordered_sessions, sig_sessions)]

        # get betas as a list....
        betas_list = []
        session_labels = []
        channel_labels = []
        area_labels = []
        num_channels = []
        for session in sig_ordered_sessions:
            channels = self.corr_obj.get_good_channels(session=session, threshold=threshold)
            session = str(int(session))
            num_channels.append(len(channels))
            if self.model_name == 'strf_model':
                session_betas = self.beta_bank[session].transpose((1,2,0))
            else:
                session_betas = self.beta_bank[session][layer].cpu().numpy()
            for ch in channels:
                ch = int(ch)
                betas_list.append(session_betas[...,ch].reshape(-1))
                session_labels.append(session)
                channel_labels.append(ch)
                area_labels.append(self.metadata.get_session_area(session))
        return betas_list, session_labels, channel_labels, area_labels
    
    def hierarchical_clustering(
            self, betas_list, metric_func=None, method = 'ward',
            no_plot=True, opt_ordering=True):
        """Performs hierarchical clustering and returns sequence of ids of ordered betas.
        
        Args:
            betas_list: list of significant betas
            metric_func: Some metric function to be used for cost.

        """
        if metric_func is None:
            metric_func = 'euclidean'

        # linkage_data = linkage(betas_list, method='ward', metric='euclidean')
        linkage_data = linkage(betas_list, method=method, metric=metric_func,
                               optimal_ordering=opt_ordering)

        R = dendrogram(linkage_data, no_plot=no_plot)
        # plt.show()
        # getting beta using clustered betas..
        cluster_ordered_ids = []
        for num in R['ivl']:
            cluster_ordered_ids.append(int(num))
        return cluster_ordered_ids, R, linkage_data
    
    def get_cluster_labels(self, list_labels, ordered_ids):
        """Takes in any list of labels (corresponsing to each beta)
        and ordered ids (result of clustering), and gives a list of
        cluster label 
        """
        cluster_labels = []
        cluster_start_ids = []
        for i, id in enumerate(ordered_ids):
            if i ==0:
                cluster_labels.append(list_labels[id])
                cluster_start_ids.append(i)
            elif list_labels[id] != cluster_labels[-1]:
                # when label changes, add it to the list..
                cluster_labels.append(list_labels[id])
                cluster_start_ids.append(i)
        # computing cluster widths..
        
        cluster_start_ids.append(len(ordered_ids))
        cluster_widths = []
        for i, id in enumerate(cluster_start_ids):
            if i != 0:
                width = id - cluster_start_ids[i-1]
                cluster_widths.append(width)

        return cluster_labels, cluster_widths
    
    def _analyzer_dendrogram(self, session_labels, ch_labels, dendrogram_R):
        """Assign clusters to the sessions and channels using dendrogram.
        Dendrogram assigns color to clusters, these assigns colors may be
        repeated. This method treats change in assigned color as a new 
        cluster and populates dictionary of dictionary, where each session
        points to a dictionary with clusters as its keys and corresponding
        channels list as values.
        
        Args:
            session_labels (list): list of session labels (prior to clustering) 
            ch_labels (list): list of channel labels (prior to clustering) 
            dendrogram_R (dict): output of scipy dendrogram plotting func.

        Returns:
            session_wise_clustering: dict of dict
            num_clusters (int)    
        """
        ordered_ids = dendrogram_R['ivl']
        assigned_colors = dendrogram_R['leaves_color_list']

        num_clusters = 0
        session_wise_clustering = {}
        cluster_wise_ids = {}
        for i, (id, color) in enumerate(zip(ordered_ids, assigned_colors)):
            id = int(id)
            if i==0:
                num_clusters = 0
                cluster_wise_ids[num_clusters] = []     # create an empty list every time new cluster starts
            elif color != last_color:
                num_clusters += 1
                cluster_wise_ids[num_clusters] = []
            
            # maintain list of ids for each cluster..
            cluster_wise_ids[num_clusters].append(id)
            session = session_labels[id]
            if session in session_wise_clustering.keys():
                if num_clusters in session_wise_clustering[session].keys():
                    session_wise_clustering[session][num_clusters].append(ch_labels[id])
                else:
                    session_wise_clustering[session][num_clusters] = [ch_labels[id]]
            else:
                session_wise_clustering[session] = {num_clusters: [ch_labels[id]]}

            last_color = color
        # since cluster indexing starts from 0, total clusters are +1 the last index
        return session_wise_clustering, num_clusters, cluster_wise_ids
    
    def get_session_wise_clustering(
            self, ordered_sessions_list, layer, cost_func,
            analysis_criteria='session-wise', opt_ordering=True
        ):
        """Returns session wise clustering info as dict of dict,
        that can be used to analyze clustering of betas, e.g.
        for making topographical pie chart.

        Args:
            ordered_sessions_list (list): 
            layer (int): 
            cost_func (function): """

        betas_list, session_labels, ch_labels, area_labels = self.get_sig_betas(
            ordered_sessions_list, layer=layer
        )
        cluster_ordered_ids, dendrogram_R, linkage_data = self.hierarchical_clustering(
            betas_list, metric_func = cost_func, method='average',
            no_plot = True, opt_ordering=opt_ordering
        )
        if analysis_criteria=='session-wise': 
            category_labels = session_labels
        else:
            dummy_labels = {'core': '200206', 'belt': '191121'}
            category_labels = []
            for area in area_labels:
                category_labels.append(dummy_labels[area])   
            # category_labels = area_labels
        session_wise_clustering, num_clusters, cluster_wise_ids = self._analyzer_dendrogram(
            category_labels, ch_labels, dendrogram_R
        )
        return session_wise_clustering, num_clusters, cluster_wise_ids, linkage_data
    
    def get_assigned_cluster(self, ordered_sessions_list, layer, metric):
        """"Returns the assigned cluster labels, in the order
        determined (ordering would match the orderering of betas_list)
        by the parameter 'ordered_sessions_list'.
        
        Args:
            ordered_sessions_list (list): ordered list sessions,
            layer (int): ID of layer to be analyzed.
        
        Returns: list of assinged clusters.
        """
            
        betas_list, session_labels, ch_labels, area_labels = self.get_sig_betas(
            ordered_sessions_list, layer=layer
        )
        # linkage_data = linkage(betas_list, method='ward', metric='euclidean')
        linkage_data = linkage(betas_list, method='average', metric=metric)
        dendrogram_R = dendrogram(linkage_data, no_plot=True)

        ordered_ids = dendrogram_R['ivl']
        assigned_colors = dendrogram_R['leaves_color_list']

        clustering_labels = np.zeros(len(ordered_ids))
        cluster_id = 0
        session_wise_clustering = {}
        for i, (id, color) in enumerate(zip(ordered_ids, assigned_colors)):
            id = int(id)
            if i==0:
                cluster_id = 0
            elif color != last_color:
                cluster_id += 1
            last_color = color
            clustering_labels[id] = cluster_id

        return clustering_labels



    
    # def plot_confusion_matrix(self, betas_list, ordered_ids=None, ax=None):
    #     """plots confusion matrix, using the betas list,
    #     if clustering=True, uses clustering_ordered_ids.
    #     Args:
    #         betas_list (list): betas in some predetermined order...
    #         clustering (bool): default=False, (depends on list_ids) 
    #         list_ids (list): if clustering is used, this will be 
    #             ordered accordingly..
    #     """
    #     num_betas = len(betas_list)
    #     beta_beta_matrix = np.zeros((num_betas, num_betas))
    #     for i, neuron_x in enumerate(ordered_ids):
    #         for j, neuron_y in enumerate(ordered_ids):
    #             beta_beta_matrix[i,j] = utils.cc_single_channel(betas_list[neuron_x], betas_list[neuron_y])

    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     image = ax.imshow(beta_beta_matrix, cmap='gray', vmin=0, vmax=1)
        
    #     return ax

    # def add_bboxes(self, box_widths, ax, lw=1, alpha=1):
    #     """Add bounding boxes on top of confusion matrix,
    #     these can be for highlighting any categories e.g
    #     individual sessions, core-belt areas and so on.

    #     Args:
    #         box_widths (list): widths of all boxes"""
    #     ## adding for rest of the sessions bboxes...
    #     origin = 0
    #     # color_options = qualitative.Paired_12.mpl_colormap
    #     total_boxes = len(box_widths)
    #     for s, width in enumerate(box_widths):
    #         color = qualitative.Paired_12.mpl_colormap(s/total_boxes)
    #         rect = patches.Rectangle(
    #             (origin, origin), width, width,
    #             linewidth = lw, color=color, fill=False, alpha=alpha
    #         )
    #         ax.add_patch(rect)    
    #         origin = origin + width

    #     return ax


    # def beta_beta_confusion_matrix(self, layer, ordered_sessions_list, clustering=False,
    #         bboxes=True, return_dict=False, ax=None, ordering_func=None
    #     ):
    #     """Plots confusion matrix in the standard form, with option of
    #     adding title and bboxes etc.
    #     """
    #     # model_name = 'wav2letter_modified'
    #     # get the list of betas for significant channels
    #     title = f"{self.model_name} \n Beta confusion matrix"

    #     betas_list, session_labels, ch_labels = self.get_sig_betas(ordered_sessions_list, layer=layer)

    #     if clustering:
    #         cluster_ordered_ids = self.hierarchical_clustering(
    #             betas_list, metric_func=self.beta_distance_metric, method='average', no_plot=True
    #             )
    #         ordered_ids = cluster_ordered_ids
    #         title += ', using clustering'
    #     else:
    #         # if clustering=False, ids are ordered simply..
    #         ordered_ids = np.arange(len(betas_list))

    #     # plotting and overlaying with bboxes (if neede)
    #     ax = self.plot_confusion_matrix(betas_list, ordered_ids=ordered_ids, ax=ax)
        
    #     if bboxes:
    #         cluster_labels, cluster_widths =  self.get_cluster_labels(session_labels, ordered_ids)
    #         self.add_bboxes(box_widths=cluster_widths, ax=ax)
    #     ax.set_title(title)

    #     if return_dict:
    #         m_data = {'ordered_ids': ordered_ids,
    #                 'session_labels': session_labels
    #                 }
    #         return ax, m_data

    #     return ax