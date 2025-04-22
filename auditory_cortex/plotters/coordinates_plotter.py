import os
import matplotlib.pyplot as plt

from auditory_cortex import results_dir
from auditory_cortex.plotters.plotter_utils import PlotterUtils
from auditory_cortex.neural_data.deprecated.neural_meta_data import NeuralMetaData
from auditory_cortex.neural_data.deprecated.recording_config import RecordingConfig

class CoordinatesPlotter:
    """"Provides functionality of plotting mini-plots at coordinate
    locations of sessions."""
    def __init__(self) -> None:
        self.m_data = NeuralMetaData(RecordingConfig)
        color_options = ['red', 'green', 'blue', 'brown']
        area_options = self.m_data.get_area_choices()
        self.area_wise_colors = {area: color for area, color in zip(area_options, color_options)}

    def get_session_color(self, session):
        """Picks the color assigned to the session based
        on neural area it came from"""
        area = self.m_data.get_session_area(session)
        return self.area_wise_colors[area]


    def _get_session_coordinates(self, session):
        """Returns the coordinates of recording site (session)."""
        return self.m_data.get_session_coordinates(session)

    def plot_background(self, ax, grid=True):
        """Plots background to be used for topographical plots"""
        plt.sca(ax)
        circle = plt.Circle((0,0),2, fill=False)
        ax.set_aspect(1)
        ax.add_artist(circle)
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xticks([-2, -1, 0 , 1, 2], [-10, -5, 0, 5, 10], rotation=0)
        ax.set_yticks([-2, -1, 0 , 1, 2], [-10, -5, 0, 5, 10])
        # ax.set_xticks([-2, 0 , 2], [-10, 0, 10], rotation=0)
        # ax.set_yticks([-2, 0 , 2], [-10, 0, 10])

        # area separating lines..
        # core-belt separation
        ax.plot([-1.5, 1.5], [-0.35, 0.65], color='k')
        # belt-parabelt separation
        ax.plot([-1.5, 1.5], [-1.15, -0.2], color='k')


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



    def scatter_sessions_for_recording_config(
            self, subject=None, annotate=False, save_tikz=False, 
            highlight_sessions=None, default_color='k',
            **kwargs
        ):
        """Makes a scatter plot of session coordinates, labelled 
        with session ID. Helps convey the locations of recording
        sites (sessions).
        
        Args:
            subject: str = subject+hemisphere out of
                    ['c_RH', 'c_RH', 'b_RH', 'f_RH']. Default=None.
                    In case of default, plots all the sessions 
            annotate: bool= annotate (label) each session scatter dot
            highlight_sessions: dict = sessions to be highlighted, with 
                corresponding color as value. e.g. {200206: orange}
            default_color: color = color to be used normally.
            
        """
        assert subject in ['c_LH', 'c_RH', 'b_RH', 'f_RH'], "Invalid recorind configuration."

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:   fig, ax = plt.subplots()

        ax = self.plot_background(ax)
        sessions_list = self.m_data.get_sessions_for_recording_config(subject)

        x_coordinates = []
        y_coordinates = []
        labels = []
        colors = []
        for session in sessions_list:
            session = int(session)
            cx, cy = self._get_session_coordinates(session)
            x_coordinates.append(cx)
            y_coordinates.append(cy)
            labels.append(session)
            if session in highlight_sessions:
                color = highlight_sessions[session]
            else:
                color = default_color

            colors.append(color)
            
            if annotate:
                ax.annotate(session, (cx-0.2, cy+0.05))

        # if subject == 'c_LH':
        #     xlabel = 'rostral - caudal (mm)'
        # else:
        #     xlabel = 'caudal - rostral (mm)'

        # c_LH have been rotated...
        xlabel = 'caudal - rostral (mm)'
        ylabel = 'ventral - dorsal (mm)'

        ax.scatter(x_coordinates, y_coordinates, c=colors)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{subject}")

        if save_tikz:
            filepath = os.path.join(results_dir, 'tikz_plots', f"recording_sites_{subject}.tex")
            PlotterUtils.save_tikz(filepath)

        return ax

    def display_session_coordinates(
            self, sessions, ax=None, annotate=True,
            sess_special_colors=None
            ):
        """Plots the given list of sessions on 2d coordinates,"""
        if ax is None:
            fig, ax = plt.subplots()
        self.plot_background(ax)
        x_coords = []
        y_coords = []
        colors = []
        for sess in sessions:
            cx, cy = self._get_session_coordinates(sess)
            x_coords.append(cx)
            y_coords.append(cy)
            if sess in sess_special_colors.keys():
                colors.append(sess_special_colors[sess])
            else:
                colors.append(self.get_session_color(sess))

            if annotate:
                ax.annotate(sess, (cx-0.2, cy+0.05))
        ax.scatter(x_coords, y_coords, color=colors)
        return ax


