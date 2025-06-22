# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging('info')

from auditory_cortex.plotters.tikzplots import plot_spectrogram_spikes_counts_and_session_coordinates


model_names = ['whisper_base', 'wav2vec2' ]
sessions = [200206, 180731] 
chs = [32, 7] # 180807, ch-1 # 180731, ch-7
# sent_ids = [218, 12]
sent_ids = [308, 56]
## 56, 212, 308
save_tikz = True
bin_width = 50
plot_spectrogram_spikes_counts_and_session_coordinates(
	model_names=model_names,
	sessions = sessions,
	chs=chs,
	bin_width=bin_width,
	sent_ids = sent_ids,
	save_tikz=save_tikz
)