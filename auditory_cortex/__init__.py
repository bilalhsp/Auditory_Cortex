# standard libraries
import os
import yaml
import numpy as np
from pycolormap_2d import ColorMap2DBremm, ColorMap2DZiegler


# select 2d color map to be used
CMAP_2D = ColorMap2DZiegler

# get parent directory of directory containing __init__.py file (parent of parent)
aux_dir = os.path.join(os.path.dirname(__file__), 'auxilliary')


# experiment configuration directory...!
config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
with open(os.path.join(config_dir, 'regression_config.yaml'), 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

results_dir = config['results_dir']
saved_corr_dir = os.path.join(results_dir, 'cross_validated_correlations')
pretrained_dir = os.path.join(results_dir, 'pretrained_weights')
opt_inputs_dir = os.path.join(results_dir, 'optimal_inputs')



# session ID's per subject and hemisphere
c_RH_sessions = np.array([190606, 190604, 190726, 190801, 180725, 180720, 180731,
                 180807, 180622, 190703, 190607, 190605, 180728, 180619, 180502])
b_RH_sessions = np.array([180405, 180501, 180719, 180808, 180627, 180814, 180810,
                 180801, 180417, 180413, 180420, 180613, 180724, 180730, 180717, 180406])
f_RH_sessions = np.array([191209, 200226, 200325, 200213, 200313, 191211, 200323,
                200312, 200219, 200401, 200318])
c_LH_sessions = np.array([200207, 191212, 191206, 200206, 191125, 200610, 191113,
                 191002, 191115, 200205, 191219, 200617, 200212, 191121, 191210])


# 2d coordinates or recording sessions...!
session_to_coordinates = {
                        190606: [-0.4,1.4], 190604: [-0.75,1.25], 190726: [-0.94,1.35],
                        190801: [-0.92,1.3], 180725: [-1.01,1.08], 180720: [-0.5,1.15],                        
                        180731: [-0.3,1.08], 180807: [0.18,0.8], 180622: [0.01,0.03],
                        190703: [-0.32,0.01], 190607: [1.1,-0.8], 190605: [0.8,-0.85],
                        180728: [0.6,-0.75], 180619: [0.35,-0.9],
                        180502: [0.25,-0.8], 
                        180405: [-0.98,1.2], 180501: [-0.65,1.05], 180719: [-0.3,1.25],
                        180808: [-1.15,1.15], 180627: [-0.8,0.98], 180814: [-0.7,0.55],
                        180810: [-0.55,0.3], 180801: [-1.25,0.2], 180417: [0.07,0.05],
                        180413: [0.4,-0.4], 180420: [-0.6,-0.65], 180613: [-0.7,-0.7],
                        180724: [-0.95,-1.2], 180730: [0.15,-0.98], 180717: [0.02,-1.1],
                        180406: [0.9,-0.96],
                        191209: [-0.55,0.98], 200226: [-0.7,0.7], 200325: [0.7,0.6],
                        200213: [0,0.25], 200313: [-0.8,0.03], 191211: [1.02,-0.08],
                        200323: [0.55,-0.5], 200312: [-1.02,-0.6], 200219: [-0.85,-0.8],
                        200401: [-0.85,-1.08], 200318: [-0.05,-1.3],
                        ########### C_LH  (manually reversed)     #####################
                        200207: [0,1.2], 191212: [0.6,0.7], 191206: [-0.92,0.95],
                        200206: [-0.92,0.60], 191125: [-0.5,0.5], 200610: [-1.08,0.2],
                        191113: [-0.5,0.02], 191002: [0.09,-0.07], 191115: [0.09,-0.5], 
                        200205: [-0.5,-0.5], 191219: [-0.92,-0.8], 200617: [-1.01,-1],
                        200212: [-0.43,-1.2], 191121: [0.07,-0.95], 191210: [0.8,-1.3]                         
                        
                        }
