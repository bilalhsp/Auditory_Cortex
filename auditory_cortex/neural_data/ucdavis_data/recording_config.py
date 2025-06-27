from dataclasses import dataclass
import numpy as np


@dataclass
class RecordingConfig:
	
    sess_wise_num_repeats = {
        'relayz_2024-10-28b_boilermaker.mat': 3,
		'relayz_2024-11-13c_boilermaker.mat': 3,
		'relayz_2024-11-15b_boilermaker.mat': 3,
        'relayz_2024-12-26b_boilermaker.mat': 12,
        'relayz_2024-12-30b_boilermaker.mat': 12,
        'relayz_2024-12-30c_boilermaker.mat': 12,
        'relayz_2025-01-09b_boilermaker.mat': 12,
        'relayz_2025-01-15b_boilermaker.mat': 12,
        'relayz_2025-01-24b_boilermaker.mat': 12,
    }
