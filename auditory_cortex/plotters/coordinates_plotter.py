import matplotlib.pyplot as plt

from auditory_cortex.neural_data.neural_meta_data import NeuralMetaData
from auditory_cortex.neural_data.recording_config import RecordingConfig

class CoordinatesPlotter:
    """"Provides functionality of plotting mini-plots at coordinate
    locations of sessions."""
    def __init__(self) -> None:
        self.m_data = NeuralMetaData(RecordingConfig)

    def _get_session_coordinates(self, session):
        """Returns the coordinates of recording site (session)."""
        return self.m_data.get_session_coordinates(session)

    def plot_background(self, ax, grid=True):
        """Plots background to be used for topographical plots"""
        plt.sca(ax)
        circle = plt.Circle((0,0),2, fill=False)
        ax.set_aspect(1)
        ax.add_artist(circle)
        ax.set_xlim([-2.0,2.0])
        ax.set_ylim([-2.0,2.0])
        plt.grid(grid)
        return ax
    
    def get_mini_axis(self, session, mini_plot_size, ax):
        """Returns axis at session specific coordinates for mini-plot"""
        cx, cy = self.m_data.get_session_coordinates(session)
        cx = (cx + 2)/4 - mini_plot_size[0]/2
        cy = (cy + 2)/4 - mini_plot_size[1]/2
        plt.sca(ax)
        mini_ax = plt.axes([cx, cy, mini_plot_size[0], mini_plot_size[1]])
        return mini_ax

    def plot_topographical(self, plotting_func, sessions_list, **kwargs):
        """Makes a topographical plot of data using 'plotting_func'
        to make inidividual plots and uses neural recording sites meta-data
        to organize the plots topographically.
        
        Args:
            plotting_func (function): a plotting function that takes in 
                plt.ax and 'session_data' to make a graph.
            sessions_list (list or ndarray): list of sessions to be plotted on
                topographical plot.

        **kwargs:
            size_mini_plot (tuple): size of each mini plot (sx, sy), default=(0.2, 0.05).
            mini_titles (bool): if True, add titles to mini plots, default=False.
            ax (plt.ax): (optional) plotting axis

            In addition to these, other kwargs can be provided for plotting_func.
        """
        if 'mini_plot_size' in kwargs:
            mini_plot_size = kwargs.pop('mini_plot_size')
        else: mini_plot_size = (0.2, 0.05)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:   fig, ax = plt.subplots()

        if 'mini_titles' in kwargs: mini_titles = kwargs.pop('mini_titles')
        else: mini_titles = False    

        ax = self.plot_background(ax)
        for session in sessions_list:
            mini_ax = self.get_mini_axis(
                session, mini_plot_size=mini_plot_size, ax=ax
                )
            if mini_titles:
                mini_ax.set_title(f"{session}")
            mini_ax.set_axis_off()
            # pass axis as keyword argument
            mini_ax = plotting_func(session, ax=mini_ax, **kwargs)



    def scatter_sessions(self, sessions_list, **kwargs):
        """Makes a scatter plot of session coordinates, labelled 
        with session ID. Helps convey the locations of recording
        sites (sessions).
        
        Args:
            sessions_list: list = list of sessions to be included 
                in the scatter plot.
        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:   fig, ax = plt.subplots()

        ax = self.plot_background(ax)

        x_coordinates = []
        y_coordinates = []
        labels = []
        for session in sessions_list:
            session = int(session)
            cx, cy = self._get_session_coordinates(session)
            x_coordinates.append(cx)
            y_coordinates.append(cy)
            labels.append(session)
            
            ax.annotate(session, (cx-0.2, cy+0.05))

        ax.scatter(x_coordinates, y_coordinates)
        ax.set_title("Coordinates of recording sites")
        return ax


