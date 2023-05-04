import os
import yaml
import auditory_cortex.analysis.config as config
from wav2letter.models import LitWav2Letter, Wav2LetterRF
import auditory_cortex.regression as Reg
import matplotlib as mpl

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



def get_wav2letter(checkpoint=None, pretrained=True):
    if pretrained:
        # dirr = os.path.dirname(os.path.abspath(__file__))
        # conf_file = 'config_rf.yaml'
        # manifest_file = os.path.join(dirr,'../../',"Wav2Letter","wav2letter","conf",conf_file)
        manifest_file = config.regression_object_paths['model_param_path']
        with open(manifest_file, 'r') as f:
            model_param = yaml.load(f, Loader=yaml.FullLoader)

        # Create model with pretrained weights....!
        if checkpoint is None:
            # pretrained_dir = model_param["results_dir"]
            # checkpoint = os.path.join(pretrained_dir, checkpoint_file)
            checkpoint = config.regression_object_paths['saved_checkpoint']
        print(f"loading weights from: {checkpoint}")
        mod = Wav2LetterRF.load_from_checkpoint(checkpoint, manifest=model_param)
    else:
        mod = Wav2LetterRF()
    return mod
    
def get_regression_obj(session='200206', model = 'wav2letter',load_features=False, checkpoint=None, pretrained=True):
    neural_data_dir = config.regression_object_paths['neural_data_dir']
    if model == 'wav2letter':
        mod = get_wav2letter(checkpoint=checkpoint, pretrained=pretrained)
    else:
        mod = model
    obj = Reg.transformer_regression(neural_data_dir, session, model=mod, load_features=load_features)
    return obj