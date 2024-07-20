import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

# Sample Data
energy_consumption = np.array([200, 220, 210, 230, 240])
occupancy = np.array([80, 85, 75, 90, 95])
temperature = np.array([22, 21, 23, 22, 21])
lighting_levels = np.array([300, 320, 280, 310, 305])
waste_generation = np.array([50, 45, 55, 60, 50])

# Fuzzy Logic for Resource Usage Prediction
energy_consumption_norm = fuzz.normalize(energy_consumption)
occupancy_norm = fuzz.normalize(occupancy)
temperature_norm = fuzz.normalize(temperature)
lighting_levels_norm = fuzz.normalize(lighting_levels)
waste_generation_norm = fuzz.normalize(waste_generation)

resource_usage = (energy_consumption_norm + occupancy_norm + temperature_norm + lighting_levels_norm) / 4

print("Normalized Resource Usage:", resource_usage)

# Plot Resource Usage
plt.plot(energy_consumption, label='Energy Consumption')
plt.plot(occupancy, label='Occupancy')
plt.plot(temperature, label='Temperature')
plt.plot(lighting_levels, label='Lighting Levels')
plt.plot(resource_usage, label='Resource Usage', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Normalized Values')
plt.title('Sustainable Workplace Environment')
plt.legend()
plt.show()
