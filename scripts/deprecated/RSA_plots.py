import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


from auditory_cortex.analyses.deprecated.rsa import RSA
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex.plotters.rsa_plotter import plot_line_with_shaded_region
from auditory_cortex.plotters.rsa_plotter import bar_plot_with_model_colors
from auditory_cortex import results_dir

# model_names = ['wav2letter_modified', 'wav2vec2',
#                 'deepspeech2', 'speech2text', 'whisper_tiny', 
#                 'whisper_base']






# ------------------  save RSA layer-wise ----------------------#

def save_RSA_layer_wise_plots(args):

    print(f"Plotting layer_wise correlations for RSA...")
    # area = 'core'
    area = args.area
    alpha = args.alpha
    identifier = 'global'
    itr = 100
    model_names = PlotterUtils.model_names
    for bin_width in args.bin_widths:
        # bin_width = 20

        
        for ind in args.inds:

            model_name = model_names[ind]

            # RSA_plot_layer_wise(model_name,
            #                     area=area, bin_width=bin_width, alpha=alpha
            #                     )
            rsa = RSA(model_name=model_name, identifier=identifier)
            corr_dict = rsa.get_layer_wise_corr(
                area=area, bin_width=bin_width, iterations=itr, size=499
            )
            plot_line_with_shaded_region(corr_dict, model_name, alpha=alpha)
            plt.title(f"RSA, {model_name}, bw-{bin_width}ms, area-{area}")
            plt.xlabel(f"Layer IDs")
            plt.ylabel(f"$\\rho$")
            plt.ylim([-0.1,0.4])

            filepath = os.path.join(results_dir, 'tikz_plots', f"RSA-layerwise-{area}-{model_name}.tex")
            PlotterUtils.save_tikz(filepath)





# ------------------  save peak RSA layer ----------------------#

def save_RSA_peak_layer_plots(args):

    print(f"Plotting peak layer vs bin_widths for RSA...")
    # alpha = 0.3
    # bin_widths = [5, 10, 20, 40, 60, 80, 100, 200, 400, 800]
    # area = 'core'
    model_names = PlotterUtils.model_names
    alpha = args.alpha
    area = args.area
    # ind = 0
    for ind in args.inds:
        model_name = model_names[ind] 
        # peak_layers = {}
        # peak_layer_means = {}
        # peak_layer_SEM = {}

        dist_for_peak_layer = {}
        # for model_name in model_names:
        rsa = RSA(model_name=model_name)
        for bin_width in args.bin_widths:
            print(f"Computing for bin-{bin_width}...")
            corr_dict = rsa.get_layer_wise_corr(
                area=area, bin_width=bin_width
            )
            layer_means = {np.mean(v):k for k,v in corr_dict.items()}
            peak_mean = max(layer_means)
            peak_layer = layer_means[peak_mean]

            dist_for_peak_layer[bin_width] = corr_dict[peak_layer]


        plot_line_with_shaded_region(dist_for_peak_layer, model_name, alpha=alpha)
        plt.title(f"RSA, peak-layer, {model_name}, area-{area}")
        plt.xlabel(f"bin widths (ms)")
        plt.ylabel(f"$\\rho$")
        plt.ylim([-0.1,0.4])
        print(f"All done, saving image...")

        filepath = os.path.join(results_dir, 'tikz_plots', f"RSA-peak-layer-{area}-{model_name}.tex")
        PlotterUtils.save_tikz(filepath)



# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description="This is to load neural spikes and cache the results on "+
        "'cache_dir' on scratch. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        '-b','--bin_widths', dest='bin_widths', nargs='+', type= int, action='store',
        default=[5, 10, 20, 40, 60, 80, 100, 200, 400, 800],
        # choices=[],
        help="List of bin_widths to cache neural data for."
    )
    parser.add_argument(
        '-m','--model', dest='inds', nargs='+', type=int, action='store',
        default=[0, 1, 2, 3, 4, 5],
        # choices=[0, 1, 2, 3, 4, 5],
        help="Select index corresponding to model."
    )

    parser.add_argument(
        '-a','--area', dest='area', type= str, action='store',
        default='core',
        choices=['core', 'belt', 'all'],
        help="Select area to make the plots for."
    )
    parser.add_argument(
        '-al','--alpha', dest='alpha', type=float, action='store',
        default=0.3,
        # choices=[0, 1, 2, 3, 4, 5],
        help="Select alpha for shaded region."
    )
    parser.add_argument(
        '-p','--peak_layer', dest='peak_layer', action='store_true',
        default=False,
        # choices=[0, 1, 2, 3, 4, 5],
        help="Set True, if want to plot peak layer vs bin_width. Default=False"
    )

    return parser


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")
    if args.peak_layer:
        save_RSA_peak_layer_plots(args)
    else:
        args.bin_widths = [20]
        save_RSA_layer_wise_plots(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")
