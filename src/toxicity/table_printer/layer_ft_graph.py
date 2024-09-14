import matplotlib.pyplot as plt
import numpy as np

# Data
layers = list(range(1, 30))  # 29 layers
edit_data = [0.8000000119, 0.8399999738, 0.6700000167, 0.7699999809, 0.8199999928, 0.8399999738, 0.8799999952, 0.7699999809, 0.9200000167, 0.8299999833, 0.9200000167, 0.8500000238, 0.8999999762, 0.8999999762, 0.8500000238, 0.8799999952, 0.9399999976, 0.8399999738, 0.9499999881, 0.9300000072, 0.8999999762, 0.9399999976, 0.9300000072, 0.9499999881, 0.9499999881, 0.9300000072, 0.8700000048, 0.9399999976, 0.9100000262]
holdout_data = [0.6000000238, 0.5699999928, 0.5799999833, 0.5199999809, 0.5699999928, 0.6000000238, 0.7099999785, 0.6200000048, 0.75, 0.7099999785, 0.7900000215, 0.7300000191, 0.7699999809, 0.8299999833, 0.75, 0.8000000119, 0.7799999714, 0.7200000286, 0.7300000191, 0.7400000095, 0.7200000286, 0.7699999809, 0.7900000215, 0.7699999809, 0.75, 0.7200000286, 0.7900000215, 0.7900000215, 0.7699999809]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the data
plt.plot(layers, edit_data, label='Edit', marker='o')
plt.plot(layers, holdout_data, label='Holdout', marker='o')

# Plot constant lines
plt.axhline(y=0.99, color='b', linestyle='--', label='All - Edit (0.99)')
plt.axhline(y=0.84, color='r', linestyle='--', label='All - Holdout (0.84)')

# Customize the plot
plt.xlabel('Layer')
plt.ylabel('Value')
plt.title('Edit and Holdout Values by Layer')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limits
plt.ylim(0, 1.1)

# Show the plot
plt.tight_layout()
plt.show()