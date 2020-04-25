# imports
import numpy as np
import dit
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from it_analyzer import Simplex_IT_analyzer

dmax = 15
tbs = 20

def get_IT(N, sw, tbs):    
    
    # read
    dirname = f"observations/MT_500_BIN_SIZE_{tbs}/"
    simplex_it = Simplex_IT_analyzer(dirname+f'observations_N_{N}_sw_{sw}_tbs_{tbs}')
    
    # compute
    stim_encodings = simplex_it.compute_stimulus_encodings()
    neuron_mis = simplex_it.compute_neuron_mutual_informations()
    neuron_Cmis = simplex_it.compute_conditional_mutual_informations()
    neuron_pids = simplex_it.compute_partial_information_decompositions()
    
    # plot
    plot_encodings_and_MI(N+1, sw, tbs, stim_encodings, neuron_mis, neuron_Cmis)
    plot_PIDs(N+1, sw, tbs, neuron_pids)
    
    return stim_encodings, neuron_mis, neuron_Cmis, neuron_pids

def get_IT_evolution(dmax, sw, tbs):
    
    # Aggregate data
    mis = [np.empty(dmax) for _ in range(3)] # one for each pair of neurons
    Cmis = [np.empty(dmax) for _ in range(3)]
    pids = [np.empty((5, dmax)) for _ in range(3)]
    
    DURATION = 1000
    I_on_bin = DURATION//tbs//4
    I_off_bin = I_on_bin*3
    for N in range(1,dmax+1):
        stim_encodings, neuron_mis, neuron_Cmis, neuron_pids = get_IT(N, sw, tbs)
        
        for i, (neuron_mi, neuron_Cmi, neuron_pair_pid) in enumerate(zip(neuron_mis, neuron_Cmis, neuron_pids)):
            mis[i][N-1] = np.mean(neuron_mi)
            Cmis[i][N-1] = np.mean(neuron_Cmi)
            pids[i][:, N-1] = np.mean(neuron_pair_pid[:, I_on_bin : I_off_bin], axis=1) # ---------- not all of it! only when I is active
            
        plt.close('all')
        
    plot_encodings_and_MI_evolution(dmax, sw, tbs, mis, Cmis)

    pids_per_pair = pids
    pids_per_type = [[pids[0][i], pids[1][i], pids[2][i]] for i in range(5)]

    plot_PIDs_evolutions_per_pair(dmax, sw, tbs, pids_per_pair)
    plot_PIDs_evolutions_per_type(dmax, sw, tbs, pids_per_type)
    plt.close('all')

def plot_encodings_and_MI(nb_neurons, sw, tbs, stim_encodings, neuron_mis, neuron_Cmis):

    simplex_dimension = nb_neurons-1
    nb_bins = len(stim_encodings[0])
    xaxis = list(np.arange(nb_bins))
    
    plt.figure(figsize=(24, 5))
    plt.subplot(131)
    plt.title('Stimulus encoding by the neurons')
    for i, encoding in enumerate(stim_encodings):
        plt.plot(xaxis, encoding, '.-', label='neuron_{}'.format(i))
    plt.xlabel(f'time_bin (stimulus is On in [{nb_bins//4}, {3*(nb_bins//4)}))')
    plt.xticks(np.arange(0,nb_bins, 2))
    plt.ylabel('mutual information between stimulus and neuron_i ')
    plt.legend()

    plt.subplot(132)
    m = max(max(neuron_mi) for neuron_mi in neuron_mis)
    plt.title('Mutual information between pairs of neurons')
    plt.plot(xaxis, neuron_mis[0], 'gs-', label='0 (source) to 1')
    plt.plot(xaxis, neuron_mis[1], 'C4d-', label='0 (source) to n (sink)')
    plt.plot(xaxis, neuron_mis[2], 'ro-', label='n-1 to n (sink)')
    plt.xlabel(f'time bin (stimulus in On in [{nb_bins//4}, {3*(nb_bins//4)}))')
    plt.xticks(np.arange(0,nb_bins, 2))
    plt.ylabel('mutual information between neuron_i and neuron_j')
    plt.ylim([0,1.5*m])
    plt.legend()
    
    plt.subplot(133)
    m = max(max(neuron_Cmi) for neuron_Cmi in neuron_Cmis)
    plt.title('Mutual information between pairs of neurons conditioned on the stimulus')
    plt.plot(xaxis, neuron_Cmis[0], 'gs-', label='0 (source) to 1')
    plt.plot(xaxis, neuron_Cmis[1], 'C4d-', label='0 (source) to n (sink)')
    plt.plot(xaxis, neuron_Cmis[2], 'ro-', label='n-1 to n (sink)')
    plt.xlabel(f'time bin (stimulus is On in [{nb_bins//4}, {3*(nb_bins//4)}))')
    plt.xticks(np.arange(0,nb_bins, 2))
    plt.ylabel('mutual information between neuron_i and neuron_j')
    plt.ylim([0,1.5*m])
    plt.legend()
    
    plt.savefig('figures/encodings_and_mi/encodings_and_mi_N_{}_sw_{}_tbs_{}.png'.format(
        simplex_dimension, sw, tbs))
    
    
def plot_dist(d):
    d.make_dense()
    fig = figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection='3d')

    x = []
    y = []

    for *neurons, pre_last, last in d.outcomes:
        x.append(pre_last)
        y.append(last)

    n = len(pre_last)
    prob = zeros(n)

    dx = 0.75*ones(n)
    dy = 0.75*ones(n)
    dprob = d.pmf

    ax1.bar3d(x, y, prob, dx, dy, dprob, shade=True)

    ax1.set_xlabel('pre_last')
    ax1.set_ylabel('last')
    ax1.set_zlabel('Pxy')
    
def plot_PIDs(nb_neurons, sw, tbs, neuron_pids):
    
    simplex_dimension = nb_neurons-1
    nb_bins = len(neuron_pids[0][0])
    xaxis = list(np.arange(nb_bins))
    
    titles = ['PID of {I, 0 (source)} & 1', 'PID of {I, 0 (source)} & n (sink)', 'PID of {I, n-1} & n (sink)']
    labels = ['total', 'synergy', 'redundancy', 'I_unique', 'neuron_unique']
    pair_markers = [[first_marker] + ['C1*-', 'C7.-', 'C9.-', 'k.-'] for first_marker in ['gs-', 'C4d-', 'ro-']]
    lws = [2.5, 2.5, 4, 1.5, 1.5]
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(221)
    m = max(max(pair_pid[0]) for pair_pid in neuron_pids)
    plt.title('Total mutual information')
    plt.plot(xaxis, neuron_pids[0][0], 'gs-', label='{I, 0 (source)} & 1')
    plt.plot(xaxis, neuron_pids[1][0], 'C4d-', label='{I, 0 (source)} & n (sink)')
    plt.plot(xaxis, neuron_pids[2][0], 'ro-', label='{I, n-1} & n (sink)')
    plt.xlabel(f'time bin (stimulus is On at [{nb_bins//4}, {3*(nb_bins//4)}))')
    plt.xticks(np.arange(0,nb_bins, 2))
    plt.ylabel('Mutual information between {I, neuron_i} & neuron_j')
    plt.ylim([0,1.5*m])
    plt.legend()
    
    for i, (pair, title, markers) in enumerate(zip(neuron_pids, titles, pair_markers)):
        plt.subplot('22'+str(i+2))
        plt.title(title)
        for info, label, marker, lw in zip(pair, labels, markers, lws):
            plt.plot(xaxis, info, marker, label=label, lw=lw, ms=3*lw)
        plt.xlabel(f'time bin (stimulus is On at [{nb_bins//4}, {3*(nb_bins//4)}))')
        plt.xticks(np.arange(0,nb_bins, 2))
        plt.ylabel('partial information decomposition')
        plt.ylim([0,1.2*m])
        plt.legend()

    plt.savefig('figures/PIDs/PIDs_N_{}_sw_{}_tbs_{}.png'.format(
        simplex_dimension, sw, tbs))
    
    
def plot_encodings_and_MI_evolution(dmax, sw, tbs, mis, Cmis):
    
    xaxis = np.arange(1,dmax+1)
    
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.title('Mutual information between pairs of neurons')
    plt.plot(xaxis, mis[0], 'gs-', label='0 (source) to 1')
    plt.plot(xaxis, mis[1], 'C4d-', label='0 (source) to n (sink)')
    plt.plot(xaxis, mis[2], 'ro-', label='n-1 to n (sink)')
    plt.xlabel('simplex dimension')
    plt.xticks(xaxis)
    plt.ylabel('mutual information between neuron_i and neuron_j')
    plt.legend()
    
    plt.subplot(122)
    plt.title('Mutual information between pairs of neurons conditioned on the stimulus')
    plt.plot(xaxis, Cmis[0], 'gs-', label='0 (source) to 1')
    plt.plot(xaxis, Cmis[1], 'C4d-', label='0 (source) to n (sink)')
    plt.plot(xaxis, Cmis[2], 'ro-', label='n-1 to n (sink)')
    plt.xlabel('simplex dimension')
    plt.xticks(xaxis)
    plt.ylabel('mutual information between neuron_i and neuron_j')
    plt.legend()
    
    plt.savefig('figures/MI_evolutions/MI_evolutions_dmax_{}_sw_{}_tbs_{}.png'.format(dmax, sw, tbs))
    
    
def plot_PIDs_evolutions_per_pair(dmax, sw, tbs, pids_per_pair):
    
    xaxis = np.arange(1,dmax+1)
    titles = ['PID of {I, 0 (source)} & 1', 'PID of {I, 0 (source)} & n (sink)', 'PID of {I, n-1} & n (sink)']
    labels = ['total', 'synergy', 'redundancy', 'I_unique', 'neuron_unique']
    pair_markers = [[first_marker] + ['C1*-', 'C7.--', 'C9.-', 'k.-'] for first_marker in ['gs-', 'C4d-', 'ro-']]
    lws = [2.5, 2.5, 4, 1.5, 1.5]
    
    plt.figure(figsize=(24, 5))
    
    for i, (pair, title, markers) in enumerate(zip(pids_per_pair, titles, pair_markers)):
        plt.subplot('13'+str(i+1))
        plt.title(title)
        for info, label, marker, lw in zip(pair, labels, markers, lws):
            plt.plot(xaxis, info, marker, label=label, lw=lw, ms=3*lw)
        plt.xlabel('simplex dimension')
        plt.xticks(xaxis)
        plt.ylabel('partial information decomposition')
        plt.legend()

    plt.savefig('figures/PID_evolution_by_pair/PID_evolution_by_pair_dmax_{}_sw_{}_tbs_{}.png'.format(dmax, sw, tbs))

def plot_PIDs_evolutions_per_type(dmax, sw, tbs, pids_per_type):
    
    xaxis = np.arange(1,dmax+1)
    titles = ['Total', 'Synergy', 'Redundancy', 'I_unique', 'Neuron_unique']
    
    plt.figure(figsize=(16, 10))
    for i, (partial_info, title) in enumerate(zip(pids_per_type, titles)):
        plt.subplot('23'+str(i+1))
        plt.title(title)
        plt.plot(xaxis, partial_info[0], 'gs-', label='{I, 0 (source)} & 1')
        plt.plot(xaxis, partial_info[1], 'C4d-', label='{I, 0 (source)} & n (sink)')
        plt.plot(xaxis, partial_info[2], 'ro-', label='{I, n-1} & n (sink)')
        plt.xlabel('simplex dimension')
        plt.xticks(xaxis)
        plt.ylabel('partial information')
        plt.legend()
        
    plt.savefig('figures/PID_evolution_by_type/PID_evolution_by_type_dmax_{}_sw_{}_tbs_{}.png'.format(dmax, sw, tbs))
    
for sw in [2,5,7, 10, 12, 15, 20, 25, 30]:
    get_IT_evolution(dmax, sw, tbs)
