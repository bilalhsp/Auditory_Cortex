import numpy as np
from auditory_cortex.neural_data.config import RecordingConfig

class NeuralMetaData:
    def __init__(self, cfg: RecordingConfig) -> None:
        self.cfg = cfg
        # session to area.
        self.session_to_area = {}
        for k, v in self.cfg.area_wise_sessions.items():
            for sess in v:
                self.session_to_area[sess] = k

    def get_session_area(self, session):
        """Returns 'area' (core/belt/PB) for the given session"""
        return self.session_to_area[int(session)]

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
        if area is None:
            sessions = []
            for k,v in self.cfg.area_wise_sessions.items():
                sessions.append(v)
            return np.concatenate(sessions)
        else:
            return self.cfg.area_wise_sessions[area]
        

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
        