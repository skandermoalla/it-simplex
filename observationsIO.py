# read and write observations:

def write_observations(observations, path_to_file):
    with open(path_to_file, 'w') as file:
        for time_bin in observations:
            file.write('new bin\n')
            for observation in time_bin:
                file.write(','.join(str(obs) for obs in observation) + '\n')

def read_observations(path_to_file):
    observations = []
    with open(path_to_file, 'r') as file:
        for line in file.readlines():
            if line.startswith('new bin'):
                observations.append([])
                continue
            observation = tuple(float(obs) for obs in line.strip().split(','))
            observations[-1].append(observation)
            
    return observations
