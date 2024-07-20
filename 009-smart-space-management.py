import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

# Sample Data
data = {
    'room_id': [1, 2, 3, 4, 5],
    'occupancy': [80, 85, 75, 90, 95],
    'environmental_conditions': [70, 75, 68, 72, 74],
    'energy_usage': [200, 220, 210, 230, 240],
    'space_reservations': [10, 12, 8, 15, 13]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define objective function for Genetic Algorithm
def objective_function(X):
    occupancy = X[0] * df['occupancy'][0] + X[1] * df['occupancy'][1] + \
                X[2] * df['occupancy'][2] + X[3] * df['occupancy'][3] + \
                X[4] * df['occupancy'][4]
    energy = X[0] * df['energy_usage'][0] + X[1] * df['energy_usage'][1] + \
             X[2] * df['energy_usage'][2] + X[3] * df['energy_usage'][3] + \
             X[4] * df['energy_usage'][4]
    return occupancy - energy

# Genetic Algorithm
varbound = np.array([[0, 1]]*5)
algorithm_param = {'max_num_iteration': 100, 'population_size': 10, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type':'uniform', 'max_iteration_without_improv':None}
model = ga(function=objective_function, dimension=5, variable_type='bool', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()
optimal_space_allocation = model.output_dict['variable']
print("Optimal Space Allocation:", optimal_space_allocation)

# Plot Space Allocation
plt.bar(df['room_id'], optimal_space_allocation, color='blue')
plt.xlabel('Room ID')
plt.ylabel('Allocated Space (0 or 1)')
plt.title('Smart Space Management System')
plt.show()
