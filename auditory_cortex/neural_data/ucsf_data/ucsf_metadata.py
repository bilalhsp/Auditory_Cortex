"""

Notes:
-04-28-2025:
    mVoc stimulus ids are complicated.
    - stimulus related information like duration, sampling rate and audio 
    are indexed by trial ids [0, 779].
    - Unique mVocs stimulus ids are [1, 303], out of these:
        + 11 are repeated 15 times, hence reserved for testing.
        + 292 have different number of presentations, mostly presented once,
            but some are presented multiple times (not exactly 15).
    - So ideally, we would like to use stimulus ids (both train and test)
        to keep track of features and spikes, but that has two problems:
        + 1. For some of the sessions, some stimuli with single presentation 
            have bad trials (no spikes against those), so we need to remove
            those trials. That leaves us with less than 200 trials for training data
            in some sessions (compared to 292 if we take all unique presentations).
        + 2. For some of the sessions, some stimuli with multiple presentations
            so we would like to use them as training data. But we cannot track them
            using stimulus ids, because they are not unique.
    - Using trial ids is not a good idea either, because it does not make sense
        to talk about ids having repeats since each trial (but stimuli are not) is unique.
    - So I am making new ids (nids) for mVocs.
        + Load and track features using trial ids.
        + For training data, nids = trial_ids
        + For testing data, nids = first_trial_id for each stim_id in all testing_stim_ids. 
    - Impact:
        + Since all stimulus data is anyways indexed by trial ids, we can use nids to get 
            stimulus info like audio, duration, sampling rate etc.
        + For extracting spikes:
            * training data can be directly extracted using trial ids.
            * For testing data...
                nids --> stim_ids --> list of trials --> get spikes for all trials
                store them in dict indexed by nids. 
"""

import os
import numpy as np
from scipy import io
from scipy.signal import resample
import fnmatch
import pickle
import wave

from auditory_cortex import neural_data_dir, NEURAL_DATASETS
# from auditory_cortex.neural_data.config import RecordingConfig
from .recording_config import RecordingConfig
from ..base_metadata import BaseMetaData, register_metadata

DATASET_NAME = NEURAL_DATASETS[0]
DATA_DIR = os.path.join(neural_data_dir, DATASET_NAME)

@register_metadata(DATASET_NAME)
class UCSFMetaData(BaseMetaData):
    def __init__(self) -> None:
    
        self.cfg = RecordingConfig()
        # session to area.
        self.session_to_area = {}
        for k, v in self.cfg.area_wise_sessions.items():
            for sess in v:
                self.session_to_area[sess] = k

        self.data_dir = DATA_DIR
        # loading stimuli metadata details...
        mat_file = 'out_sentence_details_timit_all_loudness.mat'
        self.sentences = io.loadmat(os.path.join(self.data_dir, mat_file), struct_as_record = False, squeeze_me = True, )
        self.features = self.sentences['features']
        self.phn_names = self.sentences['phnnames']
        self.sentdet = self.sentences['sentdet']
        self.sent_IDs = np.arange(1,500)
        self.test_sent_IDs = np.array([12,13,32,43,56,163,212,218,287,308])

        # mVoc metadata..
        self.mVocId_to_trialId, self.mVoc_test_stimIds, self.mVoc_test_trIds = self.get_mVoc_metadata()
        self.mVocAud, self.mVocDur, self.mVocRate = UCSFMetaData.read_mVoc_stim_details()
        self.mVocTrialIds = np.array(list(self.mVocAud.keys()))
        self.mVoc_silence_dur = 0.0 # seconds
        self.mVocs_all_stim_ids = list(self.mVocId_to_trialId.keys())

    def num_repeats_for_sess(self, sess_id, mVocs=False):
        """Returns the number of repeats (of test data) for the given session id
        
        Args:
            sess_id (int): Number of repeats for all session ids are the same,
                so this argument is not used, but kept for consistency.
            mVocs (bool): Number of repeats for mVocs if True, else TIMIT.

        Returns:
            int: Number of repeats for the given session id
        """
        stim = 'mVocs' if mVocs else 'timit'
        return self.cfg.stim_wise_num_repeats[stim]


    def total_stimuli_duration(self, mVocs=False):
        """Returns the total duration of all the stimuli in the experiment,
        separately for unique and repeated stimuli
        
        Returns:
            {'unique': float, 'repeated': float}
        """
        stim_ids = self.get_stim_ids(mVocs)
        stim_duration = {}
        for stim_type, stim_ids in stim_ids.items():
            stim_duration[stim_type] = sum([self.get_stim_duration(stim_id, mVocs) for stim_id in stim_ids])
        return stim_duration
    
    def get_stim_ids(self, mVocs=False):
        """Returns the set of stimulus ids for both unique and repeated stimuli
        Returns:
            {'unique': (n,), 'repeated': (m,)}
        """
        return {
            'unique': self.get_training_stim_ids(mVocs),
            'repeated': self.get_testing_stim_ids(mVocs),
            }

    def get_training_stim_ids(self, mVocs=False):
        """Returns the set of training stimulus ids"""
        if mVocs:
            all_trials = self.mVocTrialIds
            test_trial_ids = np.concatenate([
                self.get_mVoc_tr_id(stim_id) for stim_id in self.mVoc_test_stimIds]
                )
            train_nids = all_trials[np.isin(all_trials, test_trial_ids, invert=True)]
            return train_nids
        else:
            sent_IDs = self.sent_IDs
            testing_sent_ids = self.test_sent_IDs
            return sent_IDs[np.isin(sent_IDs, testing_sent_ids, invert=True)]

    def get_testing_stim_ids(self, mVocs=False):
        """Returns the set of testing stimulus ids"""
        if mVocs:
            # return np.array(self.mVoc_test_stimIds)
            test_trial_ids = np.array(
                [self.get_mVoc_tr_id(stim_id)[0] for stim_id in self.mVoc_test_stimIds]
                )
            return test_trial_ids
        else:
            return self.test_sent_IDs
        
    def get_stim_audio(self, stim_id, mVocs=False):
        """Return audio for stimulus (timit or mVocs) id, resampled at 16kHz"""
        if mVocs:
            # tr_id = self.get_mVoc_tr_id(stim_id)[0]
            # Since in the new system, we are dealing with nids,
            # which are actualy trial ids, we can use them directly.
            tr_id = stim_id #
            return self.get_mVoc_aud(tr_id)
        else:
            return self.stim_audio(stim_id)
        
    def get_stim_duration(self, stim_id, mVocs=False):
        """Return duration for stimulus (timit or mVocs) id"""
        if mVocs:
            # tr_id = self.get_mVoc_tr_id(stim_id)[0]
            # Since in the new system, we are dealing with nids,
            # which are actualy trial ids, we can use them directly.
            tr_id = stim_id #
            return self.get_mVoc_dur(tr_id)
        else:
            return self.stim_duration(stim_id)


    def get_sampling_rate(self, mVocs=False):
        """Returns the sampling rate of the audio stimuli."""
        if mVocs:
             return self.mVocRate
        else:
            return self.sentdet[0].soundf   #since fs is the same for all sentences, using fs for the first sentence


    def phoneme(self, sent=1):
        # subtracting 1 because timitStimcodes range [1,500) and sent indices range [0,499)
        sent -= 1
        #indices where phoneme exists
        phoneme_present = np.amax(self.sentdet[sent].phnmat, axis=0)
        # one-hot-encoding to indices (these may carry 0 where no phoneme is present)
        indices = np.argmax(self.sentdet[sent].phnmat, axis = 0)
        #eliminate 0's for no phonemes
        indices = indices[np.where(phoneme_present>0)]
        return self.phn_names[indices]
  
    def stim_audio(self, sent=1):
        """Retrieves audio waveform for given sent ID"""
        # subtracting 1 because timitStimcodes range [1,500) and sent indices range [0,499)
        sent -= 1
        fs = self.sentdet[sent].soundf
        bef, aft = 0.5, 0.5
        sound = self.sentdet[sent].sound
        sound = sound[int(bef*fs):-int(aft*fs)]
        
        return sound

    def stim_duration(self, sent=1):
        """Retrieves audio duration for given sent ID"""
        # subtracting 1 because sent_id =1 is stored at index 0 in sentdet and so on...
        sent -= 1
        bef, aft = 0.5, 0.5
        duration = self.sentdet[sent].duration - (bef + aft)
        return duration
    
    def stim_samples(self, sent, bin_width=20):
        """Returns the calculated number of samples for a sentence 
        at the specified sampling rate (bin_width).
        """
        bin_width = bin_width/1000  #ms to seconds 
        n_samples = int(np.ceil(round(self.stim_duration(sent)/bin_width, 3)))
        return n_samples
    
    def get_total_test_duration(self, mVocs=False):
        """Returns combined duration (seconds) of test set stimuli."""
        duration = 0
        for nid in self.get_testing_stim_ids(mVocs=mVocs):
            duration += self.get_stim_duration(nid, mVocs=mVocs)

        return duration

    def audio_phoneme_data(self):
        audio = {}
        phnm = {}
        for i in range(499):
            phnm[i] = self.phoneme(i)
            audio[i] = self.audio(i)
        return audio, phnm

    def get_session_area(self, session):
        """Returns 'area' (core/belt/PB) for the given session"""
        return self.session_to_area[int(session)]
    
    def get_area_choices(self):
        """Returns all brain areas covered in recordings. e.g. ['core', 'belt']"""
        area_choices = list(self.cfg.area_wise_sessions.keys())
        area_choices.append('all')
        return area_choices

    def get_session_coordinates(self, session):
        """Returns coordinates of recoring site (session)"""
        return self.cfg.session_coordinates[int(session)]
    
    def get_all_sessions(self, area=None):
        """Returns a list of all sessions, or area-specific 
        sessions.
        
        Args:
            area (str): area of auditory cortex, default=None,  
                        ('core', 'belt', 'parabelt').
        """
        if area is None or area=='all':
            sessions = []
            for k,v in self.cfg.area_wise_sessions.items():
                sessions.append(v)
            return np.sort(np.concatenate(sessions))
        else:
            return np.sort(self.cfg.area_wise_sessions[area])
        
    def get_sessions_for_recording_config(self, subject: str=None):
        """Returns sessions for the 'subject', where subject refers
        to subject+hemisphere. 
        Args:
            subject: str = subject+hemisphere out of
                    ['c_RH', 'c_RH', 'b_RH', 'f_RH']. Default=None.
                    In case of default, returns all the sessions 
        """
        if subject == 'c_LH':
            sessions = self.cfg.c_LH_sessions
        elif subject == 'c_RH':
            sessions = self.cfg.c_RH_sessions
        elif subject == 'b_RH':
            sessions = self.cfg.b_RH_sessions
        elif subject == 'f_RH':
            sessions = self.cfg.f_RH_sessions
        else:
            sessions = np.concatenate([
                    self.cfg.c_LH_sessions,
                    self.cfg.c_RH_sessions,
                    self.cfg.b_RH_sessions,
                    self.cfg.f_RH_sessions
                ])
            
        all_sessions = self.get_all_available_sessions().astype(int)
        sessions = all_sessions[np.isin(all_sessions, sessions)]
        return sessions
        

    def order_sessions_horizontally(self, reverse=False):
        """Gives a list of sessions ordered by positions along
        caudal_rostral axis (left-right)."""
        sorted_by_x_axis = dict(sorted(
            self.cfg.session_coordinates.items(),
            key=lambda item: item[1][0],
            reverse=reverse
            # ordered by x-coordinate  
        ))
        return np.array(list(sorted_by_x_axis.keys()))

    def order_sessions_vertically(self, reverse=False):
        """Gives a list of sessions ordered by positions along
        dorsal_ventral axis (top-down)"""
        sorted_by_y_axis = dict(sorted(
            self.cfg.session_coordinates.items(),
            key=lambda item: item[1][1],
            reverse=reverse
            # ordered by y-coordinate 
        ))
        return np.array(list(sorted_by_y_axis.keys()))
    
    def order_sessions_by_distance(self, session=None):
        """Gives a list of sessions ordered by distance from 
        the given session, if session=None use top-left sessions."""
        session_coordinates = self.cfg.session_coordinates
        if session is None:
            pick_left_most = 0
            max_distance = 0.00
            for k in self.get_all_sessions():
                v = self.get_session_coordinates(k)
                # print(v[0])
                distance = v[0]*v[0] + v[1]*v[1]
                if distance > max_distance and v[0] < 0 and v[1] > 0:
                    max_distance = distance
                    pick_left_most = k
            session = pick_left_most

        # sort the rest of the session by distance from the left most...
        # within core
        core_sess_distances = {}
        for sess in self.get_all_sessions('core'):
            v =  session_coordinates[sess]
            origin = self.get_session_coordinates(session)
            distance = (origin[0] - v[0])**2 + (origin[1] - v[1])**2
            core_sess_distances[sess] = distance 

        ### reverse to sort by distances
        reverse_dict ={v:k for k,v in core_sess_distances.items()}
        distances = list(reverse_dict.keys())
        distances.sort()

        sorted_distances = {i: reverse_dict[i] for i in distances}

        ### reverse to get back the session_to_distances
        core_sess_distances ={v:k for k,v in sorted_distances.items()}
        core_sessions_ordered = np.array(list(core_sess_distances.keys()))

        belt_sessions = self.get_all_sessions('belt')      
        core_belt_ordered = np.concatenate([core_sessions_ordered, belt_sessions], axis=0)
        return core_belt_ordered
    
    def get_all_available_sessions(self):
        """Retrieves the session IDs for which data is available in 'neural_data_dir'"""  
        bad_sessions = self.cfg.bad_sessions
        
        all_sessions = get_subdirectories(self.data_dir)
        sessions = all_sessions[np.isin(all_sessions, bad_sessions, invert=True)]
        sessions = np.sort(sessions.astype(str))
        return sessions
    

    def get_num_channels(self, session):
        """Returns the number of channels in a session."""
        session = str(int(float(session)))
        session_dir = os.path.join(self.data_dir, session)
        channel_filenames = np.array(os.listdir(session_dir)) 
        valid_channels = fnmatch.filter(channel_filenames,'*Ch*MUspk.mat')
        return len(valid_channels)
    

    #########################################################################
    #################  mVoc specific methods
    def get_mVoc_metadata(self):
        """Read mVoc stim codes file and returns test stim IDs, test trial IDs,
        and dictionary that gives trial Ids for stim IDs.
        """
        mat_file = 'SqMoPhys_MVOCStimcodes.mat'
        sqm_data = io.loadmat(os.path.join(self.data_dir, mat_file), struct_as_record = False, squeeze_me = True, )

        mVocId_to_trialId = {}
        mVocStimCodes = np.unique(sqm_data['mVocsStimCodes'])
        mVoc_test_stimIds = []
        mVoc_test_trIds = []

        test_trial_repetitions = 15 
        for stimCode in mVocStimCodes:
            mVocId_to_trialId[stimCode] = np.where(sqm_data['mVocsStimCodes']==stimCode)[0]
            if (mVocId_to_trialId[stimCode].size == test_trial_repetitions):
                mVoc_test_stimIds.append(stimCode)
                mVoc_test_trIds.append(mVocId_to_trialId[stimCode])

        mVoc_test_trIds = np.concatenate(mVoc_test_trIds)
        return mVocId_to_trialId, mVoc_test_stimIds, mVoc_test_trIds

    def get_mVoc_aud(self, tr_id):
        """Return mVoc aud for the tr_id, resampled at 16kHz"""
        return self.mVocAud[tr_id]
    
    
    def get_mVoc_dur(self, tr_id):
        """Return mVoc aud for the tr_id, subtracts silence
        duration from the total trial dur."""
        # silence_dur = 0.3   # seconds
        return self.mVocDur[tr_id] #- self.mVoc_silence_dur
    
    def get_mVoc_sampling_rate(self):
        """Return mVoc aud for the tr_id"""
        return self.mVocRate
    
    def get_mVoc_tr_id(self, stim_id):
        """Returns trial id for given stim_id"""
        return self.mVocId_to_trialId[stim_id]
    
    def nid_to_tr_id(self, nid):
        """Returns trial id for given nid"""
        result = next(
            ((k,v) for k, v in self.mVocId_to_trialId.items() if nid in v),    # only for test nids
                None)
        if result is None:
            raise ValueError(f"Not a valid nid: {nid}") 
        stim_id, trial_ids = result
        if trial_ids.size == 15:    # test nids
            return self.mVocId_to_trialId[stim_id]
        else:   # training nids are same as trial ids
            return np.array([nid])
        

    @staticmethod
    def extract_mVoc_stimuli_info():
        """Reads the wav file and extracts the following 
        information from the wav file,
        - trial_ID:
            - audio wavform
            - stim duration
            - stimOnset
        - sampling rate
        """
        # read wav file..
        file_name = 'MonkVocs_15Blocks.wav'
        file_path = os.path.join(DATA_DIR, file_name)
        audio_data, sampling_rate = read_wav_file(file_path)
        num_frames = audio_data.shape[0]
        # get pulse start samples
        pulse_starts,*_ = extract_pulse_info(audio_data[:,1])

        # getting the stim durations from the stimOnsets
        durations = np.diff(pulse_starts)
        # manually adding duration for last trial..
        last_duration = np.array(num_frames - pulse_starts[-1])
        durations = np.concatenate([durations, last_duration[None]])

        # convert duration in sample to duration in seconds.
        time_durations = durations/sampling_rate

        # sent wise stim waveform
        mVoc_wavforms = {}
        num_presentations = pulse_starts.shape[0]
        for i, strt in enumerate(pulse_starts):
            # last presentation
            if i == num_presentations - 1:
                mVoc_wavforms[i] = audio_data[strt:,0]
            else:
                pulse_end = pulse_starts[i+1]
                mVoc_wavforms[i] = audio_data[strt:pulse_end ,0]

        return mVoc_wavforms, time_durations, sampling_rate
    
    @staticmethod
    def write_mVoc_stim_details(new_sampling_rate=16000):
        """Extract mVoc stim details and write to disk."""
        filename = 'mVoc_stim_details.pkl'
        mVoc_filepath = os.path.join(DATA_DIR, filename)
        mVoc_wavforms, mVoc_durations, sampling_rate = UCSFMetaData.extract_mVoc_stimuli_info()
        # resample, clip silence, normalize
        mVoc_wavforms, mVoc_durations, sampling_rate = UCSFMetaData.pre_process_mVocs(
            mVoc_wavforms, sampling_rate, new_sampling_rate)

        print(f"Writing back to disk...")
        mVoc_stim_dict = {
            'stim_audios': mVoc_wavforms,
            'stim_durations':  mVoc_durations,
            'sampling_rate': sampling_rate
        }

        with open(mVoc_filepath, 'wb') as F:
            pickle.dump(mVoc_stim_dict, F)

        print(f"mVoc stim details saved to {mVoc_filepath}")

    @staticmethod
    def read_mVoc_stim_details():
        """Extract mVoc stim details and write to disk."""
        filename = 'mVoc_stim_details.pkl'
        mVoc_filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(mVoc_filepath):
            with open(mVoc_filepath, 'rb') as F:
                mVoc_stim_dict = pickle.load(F)
            return mVoc_stim_dict['stim_audios'], mVoc_stim_dict['stim_durations'], mVoc_stim_dict['sampling_rate']		
        else:
            raise FileNotFoundError(f"{mVoc_filepath} does not exist.")
        
    @staticmethod
    def pre_process_mVocs(mVoc_wavforms, sampling_rate, new_sampling_rate=16000):
        """Processes the mVoc wavforms, to get the followings:
            - Normalized amplitudes (-1, 1)
            - Resampled at 16Khz
            - Clips-off silence at the end of each trial waveform.
        Args:
            mVoc_wavforms: dict = mVoc waveforms for all 780 presentations
            sampling_rate: int = existing sampling rate of mVoc waveforms

        Returns:
            mVoc_wavforms: dict = mVoc waveforms for all 780 presentations
                with silence period clipped-off
            durations: list = duration in seconds (for all 780 presentations)
            sampling_rate: int = existing sampling rate of mVoc waveforms
        """
        # new_sampling_rate = 16000
        silence_threshold = 1e-3
        consecutive_samples = 4100	# samples worth 100 ms
        processed_mVoc_wavforms = {}
        durations = []
        print(f"Resampling at {new_sampling_rate}Hz and normalizing and clipping silence...")
        for tr, wav in mVoc_wavforms.items():

            # normalize and clip-silence...
            wav_norm = wav/30000
            sil_start = detect_silence(wav_norm, silence_threshold, consecutive_samples)
            wav_norm = wav_norm[:sil_start]

            # resample
            if new_sampling_rate != sampling_rate:
                n = int(wav_norm.size*new_sampling_rate/sampling_rate)
                wav_norm = resample(wav_norm, n)

            durations.append(wav_norm.size/new_sampling_rate)
            processed_mVoc_wavforms[tr] = wav_norm

        return processed_mVoc_wavforms, durations, new_sampling_rate
        



        


        
    
def get_subdirectories(directory):
    # List to store the names of subdirectories
    subdirectories = []
    
    # Walk through the directory
    for entry in os.scandir(directory):
        # Check if the entry is a directory
        if entry.is_dir():
            subdirectories.append(entry.name)
    return np.array(subdirectories)


def extract_pulse_info(data):
	"""Given a list containing pulses within, extracts pulse start
	points, end points and durations (in terms of samples)
	"""
	# Find where the data changes from zero to non-zero or non-zero to zero
	changes = np.diff((data != 0).astype(int))

	# Starting indices of each pulse (non-zero values)
	pulse_starts = np.where(changes == 1)[0] + 1

	# Ending indices of each pulse (non-zero values)
	pulse_ends = np.where(changes == -1)[0] + 1

	# Handle case where a pulse ends at the last element
	if data[-1] != 0:
		pulse_ends = np.append(pulse_ends, len(data))

	# Lengths of each pulse
	pulse_lengths = pulse_ends - pulse_starts

	# First trial starts with 3 onset pulses,
	#  getting rid of 2nd and 3rd pulses
	first_start = pulse_starts[0]
	pulse_starts = pulse_starts[2:]
	pulse_starts[0] = first_start

	return pulse_starts, pulse_ends, pulse_lengths


def read_wav_file(file_path):
	"""Reads the wav file and returns audio data and sampling rate"""
	# Open the WAV file
	with wave.open(file_path, 'rb') as wav_file:
		# Get the file parameters
		num_channels = wav_file.getnchannels()
		sample_width = wav_file.getsampwidth()
		frame_rate = wav_file.getframerate()
		num_frames = wav_file.getnframes()

		print(f'Frame rate (sample rate): {frame_rate} Hz')
		# Read all frames from the WAV file
		frames = wav_file.readframes(num_frames)

	# Convert the bytes data to a numpy array
	if sample_width == 1:  # 8-bit audio
		dtype = np.uint8  # unsigned 8-bit
	elif sample_width == 2:  # 16-bit audio
		dtype = np.int16  # signed 16-bit
	elif sample_width == 4:  # 32-bit audio
		dtype = np.int32  # signed 32-bit
	else:
		raise ValueError("Unsupported sample width")

	audio_data = np.frombuffer(frames, dtype=dtype)

	# If the audio has multiple channels, reshape the array to [num_frames, num_channels]
	if num_channels > 1:
		audio_data = np.reshape(audio_data, (num_frames, num_channels))

	return audio_data, frame_rate

def detect_silence(waveform, threshold, consecutive_samples=1600):
	"""Detects the start of silence and returns the index of first sample"""
	# Find the index where the silence starts
	below_threshold = np.abs(waveform) < threshold
	silence_start = np.argmax(np.convolve(below_threshold, np.ones(consecutive_samples, dtype=int), 'valid') == consecutive_samples)
	return silence_start

