from dataclasses import dataclass
import numpy as np


@dataclass
class RecordingConfig:

    stim_wise_num_repeats = {
        'timit': 11,
        'mVocs': 15,
    }


    # session ID's per subject and hemisphere
    c_RH_sessions = np.array([
        190606, 190604, 190726, 190801, 180725, 180720, 180731,
        180807, 180622, 190703, 190607, 190605, 180728, 180619, 180502
    ])
    b_RH_sessions = np.array([
        180405, 180501, 180719, 180808, 180627, 180814, 180810, 180801, 
        180417, 180413, 180420, 180613, 180724, 180730, 180717, 180406
    ])
    f_RH_sessions = np.array([
        191209, 200226, 200325, 200213, 200313, 191211,
        200323, 200312, 200219, 200401, 200318
    ])
    c_LH_sessions = np.array([
        200207, 191212, 191206, 200206, 191125, 200610, 191113, 191002,
        191115, 200205, 191219, 200617, 200212, 191121, 191210
    ])

    # 2d coordinates or recording sessions...!
    session_coordinates = {
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
        ########### C_LH  (reflected coordinates)     #####################
        200207: [0, 1.2], 191212: [0.6, 0.7], 191206: [-0.92, 0.95],
        200206: [-0.92, 0.60], 191125: [-0.5, 0.5], 200610: [-1.08, 0.2],
        191113: [-0.5, 0.02], 191002: [0.09, -0.07], 191115: [0.09, -0.5], 
        200205: [-0.5, -0.5], 191219: [-0.92, -0.8], 200617: [-1.01, -1],
        200212: [-0.43, -1.2], 191121: [0.07, -0.95], 191210: [0.8, -1.3]                                            
    }
    subject_wise_sessions = {
        'c': np.concatenate([c_RH_sessions, c_LH_sessions], axis=0),
        'b': b_RH_sessions,
        'f': f_RH_sessions
    }


    ### ref. Josh's email dated: March 07, 2024
    area_wise_sessions = {
        'core': np.array([
                # ref. Josh's email...
                # A1
                180627, 180719, 180810, 180814, 
                180720, 180731, 190604, 190606, 190726, 190801,
                191209, 200226, 200313,
                191113, 191125, 191206, 200206, 200207, 200610,
                
                180808, #moved to core..based on coordinates plot (josh was unsure)
                
                # R
                180807, 
                200213, 200325,
                191002, 191212, 

                180501  # not assigned by Josh, moved here based on coordinates plot..
            ]),
        'non-primary': np.array([

        # 'belt': np.array([
                # CL
                # 180808, moved to core..based on coordinates plot (josh was unsure)
                180405,  # close to recordings assigned to A1 (core)
                191219,

                #ML
                180613, 
                190703, 180725, 
                200219, 200312, 191115, 200205,

                # AL
                # 180717, 180730, # moved to parabelt..based on coordinates plot (josh was unsure)
                180406, 
                180622,
                191211, 200323,

                180413, 180420, # not assigned by Josh, moved here based on coordinates plot..

                # unassigned
                # 180619, #c_RH

        #     ]),            
        # 'parabelt': np.array([
                #CPB
                180724, 
                200318, 200401,
                191121, 200617,

                #RPB
                180629, #(previously not found)
                180502, 180728, 190605, 190607,
                191210,

                180717, 180730, # moved to parabelt..based on coordinates plot (josh was unsure)

                # unassigned
                # ,
                # , 180801, 180417,  
                
                200212
            ]),          
    }


    bad_sessions = [
        '200312', '200401', '191002', '180619', '180405', '180406',
        # this row (included) onwards added 01/24/24 (getting errors for inter-stimulus dead intervals)
        '180725', '191212', '200226', '200610', '200325', '191002',
        '180619', '200323', '200312',                         # belt sessions here on..
        '190607',  # SPECIAL case: doesn't give any error, but has different distribution of dead intervals..
        '180801',  # PB sessions here on...
        '180417', '200617', '200401',
    ]

    # ### Earlier area assignments (reading off the M-file)
    # area_wise_sessions = {
    #     'core': np.array([
    #             190606, 190604, 190726, 190801, 180725, 180720, 180731, 180807, #c_RH
    #             191209, 200226, 200325, 200213, 200313,   #f_RH
    #             200207, 191212, 191206, 200206, 191125, 200610, 191113, 191002, #c_LH
    #             #b_RH (need to confirm these)
    #             180405, 180719, 180808, 180627, 180814, 180810, 180724, 
                
    #             # moved to parabelt, based on comments in .m file on box..
    #             # 180406, 180717, 180613, 180730,                  
    #         ]),

    #     'belt': np.array([
    #             180622, 190703, 190607, 190605,  180619, 180502, #c_RH
    #             191211, 200323, 200312,       #f_RH
    #             191115, 200205, 191219,     #c_LH
    #             # moved to parabelt, based on comments in .m file on box..
    #             # 191210, 191121, 200617, 200212, 200219, 200401, 180728,
    #             # 200318,
    #         ]),            
    #     'parabelt': np.array([
    #             # moved to parabelt, based on comments in .m file on box..
    #             180406, 191210, 191121, 200617, 180717, 200212, 200219, 200401,
    #             180613, 180730, 180728, 200318,
    #             #b_RH (need to confirm these)
    #             180501, 180801, 180417, 180413, 180420,
    #         ]),          
    # }