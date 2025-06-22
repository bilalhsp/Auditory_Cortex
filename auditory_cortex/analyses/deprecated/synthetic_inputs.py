import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer import qualitative

# local
from auditory_cortex.optimal_input import OptimalInput
import auditory_cortex.utils as utils
from auditory_cortex.utils import SyntheticInputUtils

from auditory_cortex import opt_inputs_dir





class SyntheticInputs:
    def __init__(self, model_name, load_features=False):
        
        self.model_name = model_name
        self.wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')

        # default session for analysis
        self.session = '200206'
        self.betas = {}
        self.opt_objs = {}
        # self.colors = qualitative.Set2_8.mpl_colors
        self.colors = qualitative.Dark2_8.mpl_colors
        
        self.opt_objs[model_name] = OptimalInput(
            model_name, 
            load_features=load_features
        )
            
            # save betas using 'model+session' keys...
            # calling this will compute betas and store as attribute of opt_obj
            # _ = self.opt_objs[model_name].get_betas(self.session, use_cpu=False)

    
    def get_opt_obj(self, model_name=None):
        """Returns opt_obj for self.model_name, create one if it
        doesn't exist already."""
        if model_name is None:
            model_name = self.model_name
        if model_name not in self.opt_objs.keys():
            self.opt_objs[model_name] = OptimalInput(
                model_name, load_features=False
            )
        return self.opt_objs[model_name]
    
    def get_betas(self, session, model_name=None):
        """Compute or simply load betas (if already cached)"""
        session = str(int(session))
        if model_name is None:
            model_name = self.model_name
        beta_dir = os.path.join(opt_inputs_dir, model_name, 'betas', session)
        
        if not os.path.exists(beta_dir):
            print(f"Creating directory: {beta_dir}")
            os.makedirs(beta_dir)
        
        file_name = f"{model_name}_{session}_betas.npy"
        file_path = os.path.join(beta_dir, file_name)
        if os.path.exists(file_path):
            print("Loading betas..")
            betas = np.load(file_path)
        else:
            print("Computing betas...")
            self.opt_objs[model_name] = OptimalInput(
                model_name, load_features=True
            )
            betas = (self.get_opt_obj(model_name).get_betas(session)).cpu().numpy()
            np.save(file_path, betas)
        return betas

    def get_sessions_analyzed(self, model_name=None):
        """Returns list of sessions analyzed for the self.model_name"""
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        return os.listdir(wav_dir)
    
    def get_layers_analyzed(self, session, st_sent=None, model_name=None):
        """Returns the list of layers analyzed for the given session
        """
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name):
            layers = []
            channels = []
            sess_dir = os.path.join(wav_dir, session)
            filenames = os.listdir(sess_dir)
            
            for filename in filenames:
                # extracting starting sent
                sent_ind = filename.rfind('.wav')
                starting_sent = int(filename[sent_ind-3:sent_ind])
                
                if st_sent is not None and starting_sent != st_sent:
                    pass
                else:
                    # extracting layers and channels
                    ch_ind = filename.rfind('_corr')
                    ch = int(filename[ch_ind-2:ch_ind])
                    layer = int(filename[ch_ind-5:ch_ind-3])
                    layers.append(layer)

            return list(set(layers))

    def get_channels_analyzed(self, session, st_sent=None, layer=None, model_name=None):
        """Returns the list of channels analyzed for the given session
        """
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name):
            channels = []
            sess_dir = os.path.join(wav_dir, session)
            filenames = os.listdir(sess_dir)
            
            for filename in filenames:
                # extracting starting sent
                sent_ind = filename.rfind('.wav')
                starting_sent = int(filename[sent_ind-3:sent_ind])
                
                # extracting layers and channels
                ch_ind = filename.rfind('_corr')
                ch = int(filename[ch_ind-2:ch_ind])
                layer_analyzed = int(filename[ch_ind-5:ch_ind-3])
                if st_sent is not None and starting_sent != st_sent:
                    pass
                elif layer is not None and layer_analyzed != layer:
                    pass
                else:
                    channels.append(ch)       

            return list(set(channels))
        
    def get_st_sents_analyzed(self, session, model_name=None):
        """Returns the list of starting sents analyzed for the given session
        """
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name):
            sess_dir = os.path.join(wav_dir, session)
            filenames = os.listdir(sess_dir)
            starting_sents = []
            for filename in filenames:
                # extracting starting sent
                sent_ind = filename.rfind('.wav')
                starting_sent = int(filename[sent_ind-3:sent_ind])
                starting_sents.append(starting_sent)

            return list(set(starting_sents))
        
    def get_optimal_input(self, session, layer, ch, starting_sent, model_name=None):
        """Retrieves optimal waveform for given selection."""
        if model_name is None:
            model_name = self.model_name
        wav_dir = os.path.join(opt_inputs_dir, model_name, 'wavefiles')
        
        session = str(int(session))
        if session in self.get_sessions_analyzed(model_name): 
            sess_dir = os.path.join(wav_dir, session)
            file_found = False
            filenames = os.listdir(sess_dir)
            for filename in filenames:
                if f'{layer:02.0f}_{ch:02.0f}' in filename and f'starting_{starting_sent:03d}.wav' in filename:
                    select_file = filename
                    file_found = True
            if file_found:
                file_path = os.path.join(sess_dir, select_file)
                optimal_input = wavfile.read(file_path)
                corr_ind = select_file.rfind('_starting')
                corr = float(select_file[corr_ind-4 : corr_ind])
                return optimal_input[1], corr  
            else:
                raise FileNotFoundError(f"Optimal inputs not yet computed for layer-{layer}, ch-{ch}\
                                        with {starting_sent} as starting sent.")
        else:
            raise FileNotFoundError(f"Session-{session} not yet analyzed.")

    def plot_optimal_input(self, session, layer, ch, starting_sent, cmap='viridis', model_name=None,
                           ax=None):
        """plots spectrogram of optimal input for given selection."""
        if model_name is None:
            model_name = self.model_name
        synthetic_input, corr = self.get_optimal_input(session, layer, ch, starting_sent,
                                                       model_name=model_name)
        _, ax = SyntheticInputUtils.plot_spect(synthetic_input, ax=ax, cmap=cmap)

        ax.set_title(f" {model_name}, starting-{starting_sent} \n session-{session}, L-{layer}, ch-{ch}, $\\rho$={corr}")


    def analyze_synthetic_inputs(self, session, layer, st_sent=12,use_cpu=False, model_name=None):
        """Tabulates correlations for betas and optimal inputs for all 
        pair of channels and given layer."""

        session = str(int(session))
        # layer_idx = self.opt_objs[self.model_name].get_layer_index(layer)
        layer_idx = self.get_opt_obj(model_name).get_layer_index(layer)
        # layer_idx = self.opt_obj.linear_model.model_extractor.get_layer_index(layer)
        # betas = (self.get_opt_obj(model_name).get_betas(session)[layer_idx]).cpu().numpy()
        betas = self.get_betas(session, model_name)[layer_idx]

        df = pd.DataFrame(columns=['ch1', 'ch2', 'corr_opt_inputs','cross_corr_opt_inputs' ,'corr_betas'])
        channels = self.get_channels_analyzed(session, st_sent=st_sent, layer=layer,
                                              model_name=model_name)

        for ch1 in channels:

            opt1, _ = self.get_optimal_input(session, layer, ch1, starting_sent=st_sent,
                                             model_name=model_name)
            for ch2 in channels:
                if ch1 == ch2:
                    pass
                else:
                    opt2, _ = self.get_optimal_input(session, layer, ch2, starting_sent=st_sent, 
                                                     model_name=model_name)
                    corr_opt_inputs = utils.cc_single_channel(opt1, opt2)[0]
                    cross_corr_opt_inputs = np.max(np.correlate(opt1, opt2, mode='same'))
                    beta1 = betas[:, ch1]
                    beta2 = betas[:, ch2]
                    corr_betas = utils.cc_single_channel(beta1, beta2)[0]

                    df.loc[len(df.index)] = [ch1, ch2, corr_opt_inputs, cross_corr_opt_inputs, corr_betas]

        return df

    def plot_corr_vs_betas(self, session, layer, st_sent, ax=None, color=None,
                            model_name=None):

        if model_name is None:
            model_name = self.model_name
        if color is None:
            color = self.colors[0]
        df = self.analyze_synthetic_inputs(session, layer, st_sent, model_name=model_name)
        ax = df.plot.scatter(x='corr_betas', y='corr_opt_inputs', ax=ax, color=color)
        ax.set_title(f"{model_name}, \n session-{session}, L-{layer}, starting-{st_sent}")

    def plot_cross_corr_vs_betas(self, session, layer, st_sent, ax=None, color=None,
                                model_name=None):
        if model_name is None:
            model_name = self.model_name
        if color is None:
            color = self.colors[0]
        df = self.analyze_synthetic_inputs(session, layer, st_sent, model_name=None)
        ax = df.plot.scatter(x='corr_betas', y='cross_corr_opt_inputs', ax=ax, color=color)
        ax.set_title(f"{model_name}, \n session-{session}, L-{layer}, starting-{st_sent}")

    def analyze_synthetic_inputs_across_sessions(
            self, session1, session2, layer, st_sent=12, use_cpu=False, model_name=None
        ):
        """Tabulates correlations for betas and optimal inputs for all 
        pair of channels and given layer."""

        session1 = str(int(session1))
        session2 = str(int(session2))
        # layer_idx = self.opt_objs[self.model_name].get_layer_index(layer)
        layer_idx = self.get_opt_obj(model_name).get_layer_index(layer)
        # layer_idx = self.opt_obj.linear_model.model_extractor.get_layer_index(layer)
        # betas1 = (self.get_opt_obj(model_name).get_betas(session1)[layer_idx]).cpu().numpy()
        # betas2 = (self.get_opt_obj(model_name).get_betas(session2)[layer_idx]).cpu().numpy()
        betas1 = self.get_betas(session1, model_name)[layer_idx]
        betas2 = self.get_betas(session1, model_name)[layer_idx]
        df = pd.DataFrame(columns=['ch1', 'ch2', 'corr_opt_inputs','cross_corr_opt_inputs' ,'corr_betas'])
        channels1 = self.get_channels_analyzed(session1, st_sent=st_sent, layer=layer,
                                              model_name=model_name)
        channels2 = self.get_channels_analyzed(session2, st_sent=st_sent, layer=layer,
                                              model_name=model_name)

        for ch1 in channels1:

            opt1, _ = self.get_optimal_input(session1, layer, ch1, starting_sent=st_sent,
                                             model_name=model_name)
            for ch2 in channels2:
                opt2, _ = self.get_optimal_input(session2, layer, ch2, starting_sent=st_sent, 
                                                    model_name=model_name)
                corr_opt_inputs = utils.cc_single_channel(opt1, opt2)[0]
                cross_corr_opt_inputs = np.max(np.correlate(opt1, opt2, mode='same'))
                beta1 = betas1[:, ch1]
                beta2 = betas2[:, ch2]
                corr_betas = utils.cc_single_channel(beta1, beta2)[0]

                df.loc[len(df.index)] = [ch1, ch2, corr_opt_inputs, cross_corr_opt_inputs, corr_betas]

        return df


    def compare_sessions(self, session1, session2, layer, st_sent,
                          ax=None, model_name=None):
        if ax is None:
            fig, ax = plt.subplots()
        legend_elements = []
        # plotting session1
        self.plot_corr_vs_betas(session1, layer, st_sent, ax=ax, color=self.colors[0],
                                model_name=model_name)
        legend_elements.append(Line2D([0], [0], color=self.colors[0], label=f'{str(session1)}'))
        # plotting session1
        self.plot_corr_vs_betas(session2, layer, st_sent, ax=ax, color=self.colors[1],
                                model_name=model_name)
        legend_elements.append(Line2D([0], [0], color=self.colors[1], label=f'{str(session2)}'))
        
        # computing across sessions
        df = self.analyze_synthetic_inputs_across_sessions(
            session1, session2, layer, st_sent, model_name=None
        )
        legend_elements.append(Line2D([0], [0], color=self.colors[2], label=f'{str(session1)}-{str(session2)}'))
        # plotting across sessions..
        ax = df.plot.scatter(x='corr_betas', y='corr_opt_inputs', ax=ax, color=self.colors[2])
        ax.set_title(f"{model_name}, \n sessions-{session1} & {session2}, L-{layer}, starting-{st_sent}")
        ax.legend(handles=legend_elements, loc='best')



    def analyze_synthetic_inputs_across_models(
            self, model1, model2, session, layer1, layer2, st_sent):
        """Tabulates correlations for betas and optimal inputs for all 
        pair of channels and given layer."""

        session1 = str(int(session))

        # layer_idx = self.opt_objs[self.model_name].get_layer_index(layer)
        layer_idx1 = self.get_opt_obj(model1).get_layer_index(layer1)
        layer_idx2 = self.get_opt_obj(model2).get_layer_index(layer2)
        
        betas1 = self.get_betas(session, model1)[layer_idx1]
        betas2 = self.get_betas(session, model2)[layer_idx2]
        df = pd.DataFrame(columns=['ch1', 'ch2', 'corr_opt_inputs','cross_corr_opt_inputs' ,'corr_betas'])
        channels1 = self.get_channels_analyzed(session, st_sent=st_sent, layer=layer1,
                                              model_name=model1)
        channels2 = self.get_channels_analyzed(session, st_sent=st_sent, layer=layer2,
                                              model_name=model2)

        for ch1 in channels1:

            opt1, _ = self.get_optimal_input(session, layer1, ch1, starting_sent=st_sent,
                                             model_name=model1)
            for ch2 in channels2:
                opt2, _ = self.get_optimal_input(session, layer2, ch2, starting_sent=st_sent, 
                                                    model_name=model2)
                corr_opt_inputs = utils.cc_single_channel(opt1, opt2)[0]
                cross_corr_opt_inputs = np.max(np.correlate(opt1, opt2, mode='same'))
                beta1 = betas1[:, ch1]
                beta2 = betas2[:, ch2]
                corr_betas = utils.cc_single_channel(beta1, beta2)[0]

                df.loc[len(df.index)] = [ch1, ch2, corr_opt_inputs, cross_corr_opt_inputs, corr_betas]

        return df



    def compare_models(self, model_name1, model_name2, session, layer1, layer2, st_sent,
                          ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        legend_elements = []
        # plotting session1
        self.plot_corr_vs_betas(session, layer1, st_sent, ax=ax, color=self.colors[0],
                                model_name=model_name1)
        legend_elements.append(Line2D([0], [0], color=self.colors[0], label=f'{model_name1}'))
        # plotting session1
        self.plot_corr_vs_betas(session, layer2, st_sent, ax=ax, color=self.colors[1],
                                model_name=model_name2)
        legend_elements.append(Line2D([0], [0], color=self.colors[1], label=f'{model_name2}'))
        
        df = self.analyze_synthetic_inputs_across_models(
            model_name1, model_name2, session, layer1, layer2, st_sent)
        legend_elements.append(Line2D([0], [0], color=self.colors[2], label=f'{model_name1}-{model_name1}'))
        ax = df.plot.scatter(x='corr_betas', y='cross_corr_opt_inputs', ax=ax, color=self.colors[2])
        ax.set_title(f"{model_name1}, L-{layer1} & {model_name2}, L-{layer2} \n session-{session}, starting-{st_sent}")
        plt.legend(legend_elements)
