import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from the JSON file
file_path = "./gpu_256.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract data
frequencies = np.array([item["frequency"] for item in data])
frequencies = frequencies/1e6
execution_times = np.array([item["execution_time"] for item in data])
core_count = 256

# Amdahl's law function
def amdahl(frequency, S, T0):
    return T0 / frequency * (S + (1 - S) / core_count) 

# Fit the Amdahl model

initial_params = [0.5, execution_times.min()]

# 执行拟合
popt, pcov = curve_fit(amdahl, frequencies, execution_times, p0=initial_params)

# 拟合结果
S_opt, T0_opt = popt
print(f"Optimal S: {S_opt}, Optimal T0: {T0_opt}")

# 生成拟合曲线
fitted_times = amdahl(frequencies, S_opt, T0_opt)

# Plot real and fitted execution times
plt.figure(figsize=(10, 6))
plt.plot(frequencies, execution_times, 'o-', label="Real Execution Time", markersize=5)
plt.plot(frequencies, fitted_times, '--', label=f"Fitted Curve\n(S={S_opt:.4f}, T0={T0_opt:.4f})", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Execution Time (ms)")
plt.title("Real vs. Fitted Execution Time Using Amdahl's Law")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("Amdahl_gpu_256.png",dpi=300)

# Output fitted parameters
S_opt, T0_opt
