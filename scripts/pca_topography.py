import logging
# logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import os
import argparse
import matplotlib.pyplot as plt
from auditory_cortex.deprecated.PCA_topography import PCATopography 
# from auditory_cortex import pretrained_dir

# axis_lim = {
#         6: {0: [-2.5,1.5], 1:[-3,2], 2:[-3.5,1.5], 3:[-3,3],},
#         7: {0: [-3,3], 1:[-2,1], 2:[-2,2], 3:[-3,2.5],},
#         8: {0: [-3,2.5], 1:[-2,2], 2:[-2,1], 3:[-2,2],},
#         9: {0: [-2,1], 1:[-2,2], 2:[-2,2], 3:[-1.5,1.5],},
#         10: {0: [-2.5,0.5], 1:[-2,2], 2:[-1.5,1.5], 3:[-2,1.5],},
#             }


# creating pca_obj...
# saved_results = '/home/ahmedb/projects/Wav2Letter/saved_results/'
# pretrained_dir = '/depot/jgmakin/data/auditory_cortex/pretrained_weights/w2l_modified/'
# checkpoint_file = 'Wav2letter-epoch=024-val_loss=0.37.ckpt'
# checkpoint = os.path.join(pretrained_dir, checkpoint_file)
# pca_obj = analysis.PCA_topography()

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description="Plot PCA (KDE) plots for from good neurons (ch) from \
            all recoding session (session), using model (model_name).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
                    )

    # add arguments to read from command line
    parser.add_argument(
        'model_name', action='store', #  dest='model_name', required=True,
        choices=['wav2letter_modified', 'wav2vec', 'speech2text', 'wav2vec2'],
        help='model to be used for PCA (KDE) plots.'
    )
    parser.add_argument(
        '-t', '--threshold', dest='threshold', type=float, action='store', default=0.20, 
        help="Correlation correlation (normalizer) threshold to select neuron channels"
    )
    parser.add_argument(
        '-l','--layers', dest='layers', type=int, nargs='+', action='store', default=[6,7],
        help="Layer IDs to run the analysis for."
    )
    parser.add_argument(
        '--levels', dest='levels', type=float, nargs='+', action='store', default=[0.9],
        help="Contour levels to be plotted."
    )


    parser.add_argument(
        '-F','--threshold_factor', dest='threshold_factor', type=int, action='store', default=100,# choices=range(0, 5),
        help="Threshold factor to clip the null (kde) distribution."
    )






    # parser.add_argument(
    #     '-f','--force_redo', dest='force_redo', action='store_true', default=False, 
    #     help="set to True to recompute already existing optimal input"
    # )

    return parser


# ------------------  plot pca (kde) plots for topographic analysis ----------------------#

def plot_pca_topography(args):

    model_name = args.model_name
    layers = args.layers
    levels = args.levels
    corr_sign_threshold = args.threshold

    threshold_factor = args.threshold_factor
    normalized = True
    exclude_session = 200206.0
    # margin=0.8


    pca_obj = PCATopography(model_name=model_name)

    # setting paramters...
    # layers = [7]
    # levels = [0.9]
    # corr_sign_threshold = 0.2
    # threshold_factor=100


    if normalized:
        extend_name = '_norm'
    else:
        extend_name = ''
    for layer in layers:
        for i in range(0, 1):
            for j in range(1,3):
                if i == j or i>j:
                    continue
                pc_ind = [i,j]
                # plot_select_pcs(pc_10[pc_ind], spikes, levels=[0.9])
                ax = pca_obj.plot_significant_sessions_best_channel(
                                                layer,
                                                levels = levels,
                                                corr_sign_threshold=corr_sign_threshold,
                                                comps=pc_ind,
                                                normalized=normalized,
                                                margin=1.0,
                                                trim_axis=False,
                                                threshold_factor=threshold_factor,
                                                exclude_session=exclude_session
                                                )
                ax.set_title(f"\t PCs-{pc_ind}, layer-{layer}, threshold-{corr_sign_threshold} \
                                 \n all sessions excluding {exclude_session}")
                directory = '../../saved_results/pcs/NIPS/'
                sub_dir = os.path.join(directory, model_name)
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                plt.savefig(os.path.join(sub_dir, f'{model_name}_all_sessions_layer_{layer}_pc{pc_ind}_{extend_name}_excluding_{exclude_session}_threshold_{corr_sign_threshold}.jpg'))
                # fig_name = f"all_sessions_layer_{layer}_pc{pc_ind}_{extend_name}.tex"
                print(f"Done for layer-{layer}, pc-{pc_ind}...!")






# --------------------------------  main() ---------------------------------#


def main():

    # parge the arguments
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    # generate optimal inputs and save to the disk.
    plot_pca_topography(args)



if __name__ == "__main__":
    main()