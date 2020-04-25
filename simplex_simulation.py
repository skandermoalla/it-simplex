from brian2 import *
import observationsIO

class SimplexSimulation:
    def __init__(self, nb_neurons, synapse_weight, time_bin_size, duration):
        # prefs.codegen.target = "numpy"
        defaultclock.dt = 1*ms
        self.I_MAX = 500
        self.nb_neurons = nb_neurons
        self.synapse_weight = synapse_weight
        self.time_bin_size = time_bin_size
        self.duration = duration
        self.nb_bins = duration//time_bin_size
        
        # Neurons
        neuron_namespace = {
            'synapse_weight': synapse_weight,
            'I_MAX': self.I_MAX,
            'I_on': TimedArray([0, self.I_MAX, self.I_MAX, 0], dt=(duration//4)*ms),
            'I_weak': TimedArray([0, self.I_MAX/2, self.I_MAX/2, 0], dt=(duration//4)*ms),
            'I_off': TimedArray([0], dt=duration*ms),
            'a': 0.03, 'b': -2, 'c': -50, 'd': 100,
            'vr': -60, 'vt': -40, 'vpeak': 35,
            'C': 100, 'k': 0.7, 'tau': 1*ms,
            'input_func': TimedArray([0], dt=duration*ms)}
        model =\
        '''
        dv/dt = (k*(v-vr)*(v-vt) - u + I)/(C*tau) + 5*xi*sqrt(1/tau): 1
        du/dt = a*(b*(v - vr) - u)/tau : 1
        I = input_func(t)*(i==0) : 1          # only first neuron
        nb_spikes_in_bin : 1
        '''
        reset =\
        '''
        nb_spikes_in_bin += 1
        v = c
        u += d
        '''
        peak_threshold = 'v>vpeak' 
        self.neurons = NeuronGroup(self.nb_neurons, model, threshold=peak_threshold, reset=reset,
                         method='euler', namespace=neuron_namespace)
        self.neurons.v = self.neurons.namespace['vr']
        self.neurons.u = self.neurons.namespace['b']*self.neurons.v
        
        
        # Synapses
        synapse_namespace = {'synapse_weight': synapse_weight}
        self.S = Synapses(self.neurons, self.neurons, on_pre='v_post += synapse_weight', namespace=synapse_namespace)
        self.S.connect(condition='i < j')
        
        
        # Network
        self.spikemon = SpikeMonitor(self.neurons, record=True)
        self.spikemon.active = False
        
        @network_operation(dt=time_bin_size*ms)
        def update_time_bin(t):
            if t/ms == 0:
                return
            
            obs = tuple([self.neurons.namespace['input_func'](t-defaultclock.dt)] + list(self.neurons.nb_spikes_in_bin))
            bin_index = int((t/ms)/time_bin_size)
            self.observations[bin_index-1].append(obs)
            self.neurons.nb_spikes_in_bin = 0
        
        
        self.network = Network(self.neurons, self.S, update_time_bin, self.spikemon)
        self.network.store()
        
    def simulate(self, N_monte_carlo, path_to_dir):
        self.observations = [[] for bin_index in range(self.nb_bins)]
        
        for _ in range(N_monte_carlo):
            self.run_once('I_on')
            
        for _ in range(N_monte_carlo):
            self.run_once('I_off')
            
        observationsIO.write_observations(
            self.observations, path_to_dir+'observations_N_{}_sw_{}_tbs_{}'.format(self.nb_neurons-1, self.synapse_weight, self.time_bin_size))
    
    def run_once(self, input_power):
        self.network.restore()
        self.neurons.namespace['input_func'] = self.neurons.namespace[input_power]
        self.network.run(self.duration*ms + defaultclock.dt)
    
    
    def run_and_plot_example_raster(self, path_to_file):
        self.observations = [[] for bin_index in range(self.nb_bins)]
        self.spikemon.active = True
        
        self.run_once('I_on')
        self.plot_raster('I_on', path_to_file)
        
        self.run_once('I_off')
        self.plot_raster('I_off', path_to_file)
        
        self.spikemon.active = False
          
    def plot_raster(self, input_power, path_to_dir):
        plt_title = 'neuron_raster_N_{}_sw_{}_tbs_{}_when_{}'.format(self.nb_neurons-1, self.synapse_weight, self.time_bin_size, input_power)
        fig = figure(figsize=(15, 4))
        title(plt_title)
        plot(self.spikemon.t/ms, self.spikemon.i, '.k')
        xlabel('Time (ms)')
        ylabel('Neuron index')
        yticks(arange(self.nb_neurons))
        for t in arange(0, self.duration + 1, self.time_bin_size):
            axvline(t, ls='--', c='C1', lw=1)
        savefig(path_to_dir+plt_title+'.png')
        plt.close('all')