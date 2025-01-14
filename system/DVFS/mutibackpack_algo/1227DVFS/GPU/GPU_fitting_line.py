import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

batch = 512
file_path = f"gpu_frequency_time_energy_{batch}.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract data
frequencies = np.array([item["frequency"] for item in data])
frequencies = frequencies
execution_times = np.array([item["execution_time"] for item in data])


# 定义修正模型函数
Tmin = min(execution_times)
maxFreq = max(frequencies)
def adjusted_model(frequencies, alpha):
    return Tmin * (maxFreq / frequencies)**alpha  

# 拟合数据
popt, _ = curve_fit(adjusted_model, frequencies, execution_times, p0=[1])

# 获取拟合参数
alpha = popt
print("alpha", alpha)
# alpha
# 计算拟合曲线
fitted_time = adjusted_model(frequencies, *popt)

# 可视化对比
plt.plot(frequencies, execution_times, 'o-', label="Observed Time")
plt.plot(frequencies, fitted_time, '--', label="Fitted Model")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Execution Time (s)")
plt.title("Adjusted Execution Time Model")
plt.legend()
# plt.show()


plt.savefig(f"GPU_fitting_line_{batch}.png",dpi=300)
