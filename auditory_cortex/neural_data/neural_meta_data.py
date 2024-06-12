import os
import numpy as np
from scipy import io
import fnmatch

from auditory_cortex import neural_data_dir, config
# from auditory_cortex.neural_data.config import RecordingConfig
from auditory_cortex.neural_data import recording_config

class NeuralMetaData:
    def __init__(self, cfg: recording_config.RecordingConfig = None) -> None:
        
        if cfg is None:
            cfg = recording_config.RecordingConfig()
        self.cfg = cfg
        # session to area.
        self.session_to_area = {}
        for k, v in self.cfg.area_wise_sessions.items():
            for sess in v:
                self.session_to_area[sess] = k

        # loading stimuli metadata details...
        mat_file = 'out_sentence_details_timit_all_loudness.mat'
        self.sentences = io.loadmat(os.path.join(neural_data_dir, mat_file), struct_as_record = False, squeeze_me = True, )
        self.features = self.sentences['features']
        self.phn_names = self.sentences['phnnames']
        self.sentdet = self.sentences['sentdet']
        self.sent_IDs = np.arange(1,500)
        self.test_sent_IDs = np.array([12,13,32,43,56,163,212,218,287,308])

    def get_sampling_rate(self):
        """Returns the sampling rate of the audio stimuli."""
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
        #bef = self.sentdet[sent].befaft[0]
        #aft = self.sentdet[sent].befaft[1]
        sound = self.sentdet[sent].sound
        sound = sound[int(bef*fs):-int(aft*fs)]
        
        return sound

    def stim_duration(self, sent=1):
        """Retrieves audio duration for given sent ID"""
        # subtracting 1 because sent_id =1 is stored at index 0 in sentdet and so on...
        sent -= 1
        bef, aft = 0.5, 0.5
        #bef = self.sentdet[sent].befaft[0]
        #aft = self.sentdet[sent].befaft[1]
        duration = self.sentdet[sent].duration - (bef + aft)
        return duration
    
    def stim_samples(self, sent, bin_width=20):
        """Returns the calculated number of samples for a sentence 
        at the specified sampling rate (bin_width).
        """
        bin_width = bin_width/1000  #ms to seconds 
        n_samples = int(np.ceil(round(self.stim_duration(sent)/bin_width, 3)))
        return n_samples
    
    def get_total_test_duration(self):
        """Returns combined duration (seconds) of test set stimuli."""
        duration = 0
        for sent in self.test_sent_IDs:
            duration += self.stim_duration(sent)
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
        bad_sessions = config['bad_sessions']
        sessions = np.array(os.listdir(neural_data_dir))
        sessions = np.delete(sessions, np.where(sessions == "out_sentence_details_timit_all_loudness.mat"))
        for s in bad_sessions:
            sessions = np.delete(sessions, np.where(sessions == s))
        sessions = np.sort(sessions)
        return sessions
    

    def get_num_channels(self, session):
        """Returns the number of channels in a session."""
        session = str(int(float(session)))
        session_dir = os.path.join(neural_data_dir, session)
        channel_filenames = np.array(os.listdir(session_dir)) 
        valid_channels = fnmatch.filter(channel_filenames,'*Ch*MUspk.mat')
        return len(valid_channels)
        