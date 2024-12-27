import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from the JSON file
file_path = "./frequency&power.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract data
frequencies = np.array([item["frequency"] for item in data])
frequencies = frequencies / 1e6  # Convert to MHz for easier comparison
execution_times = np.array([item["execution_time"] for item in data])
core_count = 4

# Fix T0 as the maximum frequency's execution time (so that the fitted curve matches at max frequency)
max_frequency_idx = np.argmax(frequencies)  # Find index of the max frequency
T0_fixed = execution_times[max_frequency_idx]  # Corresponding execution time at max frequency

# Amdahl's law function (only S is variable)
def amdahl_fixed_t0(frequency, S):
    return T0_fixed / frequency * (S + (1 - S) / core_count)

# Initial parameter for S
initial_S = 0.5

# Fit the Amdahl model
popt, pcov = curve_fit(amdahl_fixed_t0, frequencies, execution_times, p0=[initial_S])

# Fitted result for S
S_opt = popt[0]
print(f"Optimal S: {S_opt}, Fixed T0: {T0_fixed}")

# Generate fitted curve
fitted_times = amdahl_fixed_t0(frequencies, S_opt)

# Plot real and fitted execution times
plt.figure(figsize=(10, 6))
plt.plot(frequencies, execution_times, 'o-', label="Real Execution Time", markersize=5)
plt.plot(frequencies, fitted_times, '--', label=f"Fitted Curve\n(S={S_opt:.4f}, T0={T0_fixed:.4f})", linewidth=2)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Execution Time (ms)")
plt.title("Real vs. Fitted Execution Time Using Amdahl's Law (T0 Fixed)")
plt.legend()
plt.grid(True)
plt.savefig("Amdahl_fixed_T0.png", dpi=300)

# Output fitted parameters
print(f"Final Fitted Parameters: S={S_opt}, T0 (fixed)={T0_fixed}")
