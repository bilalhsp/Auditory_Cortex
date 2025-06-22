import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from palettable.colorbrewer import qualitative
from auditory_cortex import utils
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex.analyses.deprecated.regression_weights import BetaAnalyzer
from auditory_cortex.neural_data.deprecated.neural_meta_data import NeuralMetaData
from auditory_cortex.neural_data.deprecated.recording_config import RecordingConfig
from auditory_cortex.plotters.coordinates_plotter import CoordinatesPlotter
from auditory_cortex.analyses.deprecated.rsa import RSA

from auditory_cortex.plotters.hierarchical_plotter import Tree, linkage_to_edges

class BetaPlotter():
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.beta_analyzer = BetaAnalyzer(model_name)
        self.metadata = NeuralMetaData(RecordingConfig)
        self.colormap = qualitative.Paired_12.mpl_colormap
        # self.score_func = utils.cc_single_channel
        self.score_func = RSA.compute_similarity
        self.corr_cost = lambda x,y: 1- self.score_func(x,y) 
        
    def _plot_confusion_matrix(self, betas_list, ordered_ids=None,
                    score_func=None, ax=None, cmap='gray'):
        """plots confusion matrix, using the betas list, uses ordered_ids
        to cluster the 'betas' according to some order.
        Args:
            betas_list (list): betas in some predetermined order...
            score_func (function): metric to be used for creating
                confusion matrix, it needs to be a functions that
                takes in two betas and computes cost. Default = None.
            ordered_ids (list): list of id, to be used for betas ordering
            ax (plt.ax): matplotlib axis to be used for plotting.
        """
        if score_func is None:
            score_func = self.score_func

        num_betas = len(betas_list)
        beta_beta_matrix = np.zeros((num_betas, num_betas))
        # if no ordering provided, use simple ordering..
        if ordered_ids is None:
            ordered_ids = np.arange(num_betas)
        for i, neuron_x in enumerate(ordered_ids):
            for j, neuron_y in enumerate(ordered_ids):
                beta_beta_matrix[i,j] = score_func(
                    betas_list[neuron_x], betas_list[neuron_y]
                )

        # if ax is None:
        #     fig, ax = plt.subplots()
        image = ax.imshow(beta_beta_matrix, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(image, ax=ax, label='corr strenth')
    
    def add_bboxes(self, box_widths, ax, lw=1, alpha=1):
        """Add bounding boxes on top of confusion matrix,
        these can be for highlighting any categories e.g
        individual sessions, core-belt areas and so on.

        Args:
            box_widths (list): widths of all boxes"""
        ## adding for rest of the sessions bboxes...
        origin = 0
        # color_options = qualitative.Paired_12.mpl_colormap
        total_boxes = len(box_widths)
        for s, width in enumerate(box_widths):
            color = self.colormap(s/total_boxes)
            rect = patches.Rectangle(
                (origin, origin), width, width,
                linewidth = lw, color=color, fill=False, alpha=alpha
            )
            ax.add_patch(rect)    
            origin = origin + width

        return ax

    def plot_confusion_matrix(self, layer, ordered_sessions_list=None,
            clustering=False, bboxes=True, return_dict=False, ax=None,
            bboxes_labels='sessions', cmap='gray'
        ):
        """Plots confusion matrix in the standard form, with option of
        adding title and bboxes etc.
        """
        # model_name = 'wav2letter_modified'
        # get the list of betas for significant channels
        
        title = f"{self.model_name} \n Beta confusion matrix"
        if ordered_sessions_list is None:
            ordered_sessions_list = self.metadata.order_sessions_horizontally()
        
        betas_list, session_labels, ch_labels, area_labels = self.beta_analyzer.get_sig_betas(
            ordered_sessions_list, layer=layer
        )
        if clustering:
            cluster_ordered_ids, _ = self.beta_analyzer.hierarchical_clustering(
                betas_list, metric_func=self.corr_cost, method='average',
                no_plot=True
            )
            ordered_ids = cluster_ordered_ids
            title += ', using clustering'
        else:
            # if clustering=False, ids are ordered simply..
            ordered_ids = np.arange(len(betas_list))


        if ax is None:
            fig, ax = plt.subplots()
        # plotting and overlaying with bboxes (if neede)
        self._plot_confusion_matrix(
            betas_list, ordered_ids=ordered_ids, ax=ax, cmap=cmap
            )    
        if bboxes:
            if bboxes_labels != 'sessions':
                box_labels = area_labels
                title += '\nboxes with area labels'
            else:
                box_labels = session_labels
                '\nboxes with session labels'
            box_labels, box_widths = self.beta_analyzer.get_cluster_labels(
                box_labels, ordered_ids
            )
            self.add_bboxes(box_widths=box_widths, ax=ax)
        ax.set_title(title)

        if return_dict:
            m_data = {'ordered_ids': ordered_ids,
                    'session_labels': session_labels
                    }
            return ax, m_data
        return ax
    

    def plot_dendrogram(self, layer, ordered_sessions_list=None, opt_ordering=True):
        """Plots dendrogram based on hiererchical clustering...
        
        Args:
            layer (int): ID of the layer to be analyzed.
            ordered_sessions_list (list): sessions in some 
                pre-determined ordering.

        """
        title = f"{self.model_name}, L-{layer}"
        if ordered_sessions_list is None:
            ordered_sessions_list = self.metadata.order_sessions_horizontally()
        
        betas_list, session_labels, ch_labels, area_labels = self.beta_analyzer.get_sig_betas(
            ordered_sessions_list, layer=layer
        )
        cluster_ordered_ids, *_ = self.beta_analyzer.hierarchical_clustering(
            betas_list, metric_func=self.corr_cost, method='average',
            no_plot=False, opt_ordering=opt_ordering
        )
        plt.title(title)


    def _plot_session_pie_chart(self, session, **kwargs):
        """Makes pie chart for session, given the session_wise_cluster,
        
        Args:
            session: neural data session id
            
        **kwargs:
            session_wise_clustering (dict of dict): this should contain 
                dictionary for each session of the form: 
                {
                    session: {
                           cluster_id: list of channels in this cluster      
                            }
                } 
            ax (plt.ax): matplotlib axis.
        """
        if 'session_wise_clustering' in kwargs:
            session_wise_clustering = kwargs.pop('session_wise_clustering')
        else:
            print(f"Keyword argument 'session_wise_clustering' must be provided.")
        if 'num_clusters' in kwargs:
            num_clusters = kwargs.pop('num_clusters')
        else:
            print(f"Keyword argument 'num_clusters' must be provided.")
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fig, ax = plt.subplots()
        if 'cmap' in kwargs:
            cmap = kwargs.pop('cmap')
            if not isinstance(cmap, mpl.colors.ListedColormap):
                cmap = plt.get_cmap(cmap)
        else:
            cmap = self.colormap
        if 'split_cmap' in kwargs:
            split_cmap = kwargs.pop('split_cmap')
        else:
            split_cmap = True
        
        if len(kwargs) != 0:
            raise ValueError('''You have provided unrecognizable keyword args''')

        session = str(int(session))
        pie_data = session_wise_clustering[session]
        cluster_wise_channels = []
        labels = []
        colors = []

        for cluster, channels_list in pie_data.items():
            cluster_wise_channels.append(len(channels_list))
            labels.append(str(cluster))
            if split_cmap:
                transformed_arg = PlotterUtils.pre_cmap_transform(cluster/num_clusters)
            else:
                transformed_arg = cluster/num_clusters
            colors.append(cmap(transformed_arg))

        ax.pie(cluster_wise_channels, labels=labels, colors=colors)
        return ax

        
    def plot_topographical_pie_chart(
            self, layer, ordered_sessions_list=None, analysis_criteria='session-wise',
            ax=None, add_colorbar=True, opt_ordering=True, **kwargs
        ):
        """Makes topographical plot with pie charts for 
        each session at their respective coordinates.
        
        Args:
            layer (int): ID of the layer to be analyzed.
        """
        # do not pop the parameter, instead pass it on
        # to the plotting function.
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
            if not isinstance(cmap, mpl.colors.LinearSegmentedColormap):
                cmap = plt.get_cmap(cmap)
        else:
            cmap = self.colormap
        if 'split_cmap' in kwargs:
            split_cmap = kwargs['split_cmap']
        else:
            split_cmap = False

        if ax is None:
            fig, ax = plt.subplots()
        # clustering...
        if ordered_sessions_list is None:
            print(f"Getting horizontally ordered sessions...")
            ordered_sessions_list = self.metadata.order_sessions_horizontally()
        session_wise_clustering, num_clusters, cluster_wise_ids, linkage_data=self.beta_analyzer.get_session_wise_clustering(
            ordered_sessions_list, layer, self.corr_cost, analysis_criteria=analysis_criteria,
            opt_ordering=opt_ordering
        )
        
        # pax = PlotterUtils.add_color_bar_with_splits(
        #     num_clusters, ax, cmap=cmap, c_label="clusters",
        #     split_levels=2, plain=not split_cmap
        #     )
        cmap = self.get_cluster_wise_colorbar(cluster_wise_ids, linkage_data)
        kwargs['cmap'] = cmap
        if add_colorbar:
            # creates space in the provided axis and adds colormap
            pax = PlotterUtils.add_color_bar(
                ax, cmap, 
                c_label="clusters",
                boundaries = np.arange(num_clusters),
                values = np.arange(num_clusters-1)
            ) 
        else:
            pax = ax
        # use coordinate plotter to make topographical plot.
        sessions = list(session_wise_clustering.keys())
        plotter = CoordinatesPlotter()
        plotter.plot_topographical(
            plotting_func=self._plot_session_pie_chart,
            sessions_list=sessions,
            ax=pax,
            session_wise_clustering=session_wise_clustering,
            num_clusters=num_clusters,
            **kwargs
        )
        pax.set_title(f"{self.model_name}, L-{layer}")

        return ax, num_clusters


    def get_cluster_wise_colorbar(self, cluster_wise_ids, linkage_data):
        """Returns hierarchical colormap based on tree colors,
        takes into account the hierarchical structure of clustering.
        """
        edges, root_label = linkage_to_edges(linkage_data=linkage_data)
        self.tree = Tree(edges=edges, root=root_label)
        r = (0, 360)
        f = 0.95
        B_l=-5
        L_1=70
        B_c=5
        C_1=60
        self.tree.assign_HCL(r,f,
            B_l=B_l, L_1=L_1,
            B_c=B_c, C_1=C_1
            )
        # cluster_based_hls = {}
        # for cluster, ids in cluster_wise_ids.items(): 
        #     hues_list = []
        #     luminance_list = []
        #     chroma_list = []
        #     rgb_list = []
        #     for id in ids:
        #         hues_list.append(self.tree.get_node_attribute(id, 'hue'))
        #         luminance_list.append(self.tree.get_node_attribute(id, 'luminance'))
        #         chroma_list.append(self.tree.get_node_attribute(id, 'chroma'))
        #         rgb_list.append(self.tree.get_node_attribute(id, 'rgb_color'))

        #     cluster_based_hls[cluster] = {'hue': hues_list,
        #                                     'luminance': luminance_list,
        #                                     'chroma': chroma_list,
        #                                     'rgb': rgb_list}

        # hls_color_list = []
        # rgb_color_list = []
        # for cluster, colors in cluster_based_hls.items():
        #     h = np.mean(colors['hue'])
        #     l = np.mean(colors['luminance'])
        #     s = np.mean(colors['chroma'])
        #     # print(h)
        #     hls_color_list.append([h,l,s])
        #     rgb_color_list.append(colorsys.hls_to_rgb(h,l,s))

        # cmap = PlotterUtils.create_cmap_using_hsl(hls_color_list)
        cluster_based_hls = {}
        hls_color_list = []
        for cluster, ids in cluster_wise_ids.items():
            cluster_depth = self.tree.get_node(ids[0]).get_attribute('depth')
            cluster_representative_child = ids[0]
            for id in ids:
                depth = self.tree.get_node(id).get_attribute('depth')
                if depth < cluster_depth:
                    cluster_depth = depth
                    cluster_representative_child = id
            cluster_root_label = self.tree.get_parent_node_label(cluster_representative_child)
            hls_color = self.tree.get_node(cluster_root_label).get_attribute('hls_color')
            cluster_based_hls[cluster] = hls_color
            hls_color_list.append(hls_color)          
        cmap = PlotterUtils.create_cmap_using_hsl(hls_color_list)

        return cmap      
