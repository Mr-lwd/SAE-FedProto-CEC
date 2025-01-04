import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


iter = "4th"
# Load the data from the JSON file
file_path = f"./MNIST_{iter}_model_DVFS.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract data
frequencies = np.array([item["frequency"] for item in data])
frequencies = frequencies/1e6
execution_times = np.array([item["execution_time"] for item in data])

T0 = min(execution_times)
maxfreq = max(frequencies)
# Amdahl's law function
def get_theory_times(frequency, maxfreq, T0):
    return T0 * (maxfreq / frequency)

# 生成拟合曲线
Theory_times = get_theory_times(frequencies, maxfreq, T0)

# Plot real and fitted execution times
plt.figure(figsize=(10, 6))
plt.plot(frequencies, execution_times, 'o-', label="Real Execution Time", markersize=5)
plt.plot(frequencies, Theory_times, '--', label="Theory Line", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Execution Time (s)")
plt.title("Real vs. Theory Execution Time")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(f"Theory_line_{iter}.png",dpi=300)


