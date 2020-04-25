from simplex_simulation import SimplexSimulation

N_MONTE_CARLO = 100 # 500
TIME_BIN_SIZE = 50
PATH_TO_DIR = f'observations/MT_{N_MONTE_CARLO}_BIN_SIZE_{TIME_BIN_SIZE}/'
DURATION = 1000
SINAPSE_WEIGHTS = [7,10, 12] # [2,5,7,10,12,15,20,25,30]
MAX_DIM = 15


def simulate_all():
    for synapse_weight in SINAPSE_WEIGHTS:
        print(f'Synapse weight == {synapse_weight}')
        for dimension in range(1, MAX_DIM+1):
            print(f'Dimension == {dimension}')
            simulation = SimplexSimulation(dimension+1, synapse_weight, TIME_BIN_SIZE, DURATION)
            simulation.run_and_plot_example_raster(PATH_TO_DIR+'raster/')
            simulation.simulate(N_MONTE_CARLO, PATH_TO_DIR)
            
            
if __name__ == '__main__':
    simulate_all()