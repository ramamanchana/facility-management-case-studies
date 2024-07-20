import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

# Sample Data
data = {
    'equipment_id': [1, 2, 3, 4, 5],
    'energy_consumption': [200, 220, 210, 230, 240],
    'maintenance_history': [2, 3, 1, 2, 3],
    'environmental_impact': [50, 45, 55, 60, 50]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define objective function for Genetic Algorithm
def objective_function(X):
    energy = X[0] * df['energy_consumption'][0] + X[1] * df['energy_consumption'][1] + \
             X[2] * df['energy_consumption'][2] + X[3] * df['energy_consumption'][3] + \
             X[4] * df['energy_consumption'][4]
    impact = X[0] * df['environmental_impact'][0] + X[1] * df['environmental_impact'][1] + \
             X[2] * df['environmental_impact'][2] + X[3] * df['environmental_impact'][3] + \
             X[4] * df['environmental_impact'][4]
    return energy + impact

# Genetic Algorithm
varbound = np.array([[0, 1]]*5)
algorithm_param = {'max_num_iteration': 100, 'population_size': 10, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type':'uniform', 'max_iteration_without_improv':None}
model = ga(function=objective_function, dimension=5, variable_type='bool', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()
optimal_schedule = model.output_dict['variable']
print("Optimal Maintenance Schedule:", optimal_schedule)

# Plot Maintenance Schedule
plt.bar(df['equipment_id'], optimal_schedule, color='green')
plt.xlabel('Equipment ID')
plt.ylabel('Scheduled Maintenance (0 or 1)')
plt.title('Sustainable Maintenance Practices')
plt.show()
