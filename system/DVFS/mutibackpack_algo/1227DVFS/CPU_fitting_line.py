import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

iters = ["1st","2nd","3rd","4th","5th","6th","7th", "8th", "9th"]
# iters = ["8th"]
def cpu_plot(iter):
    file_path = f"./MNIST_{iter}_model_DVFS.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract data
    frequencies = np.array([item["frequency"] for item in data])
    frequencies = frequencies
    execution_times = np.array([item["all_time"] for item in data])


    # 定义修正模型函数
    Tmin = execution_times[0]
    maxFreq = max(frequencies)
    def adjusted_model(frequencies, alpha):
        return Tmin * (maxFreq / frequencies)**(alpha)

    # 拟合数据
    popt, _ = curve_fit(adjusted_model, frequencies, execution_times)

    # 获取拟合参数
    alpha = popt
    print("alpha", alpha)
    # alpha
    # 计算拟合曲线
    # popt=[0.8]
    fitted_time = adjusted_model(frequencies, *popt)

    plt.figure()
    # 可视化对比
    plt.plot(frequencies, execution_times, 'o-', label="Observed Time")
    plt.plot(frequencies, fitted_time, '--', label="Fitted Model")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Execution Time (s)")
    plt.title("Adjusted Execution Time Model")
    plt.legend()
    # plt.show()


    plt.savefig(f"CPU_fitting_line_{iter}.png",dpi=300)

for iter in iters:
    cpu_plot(iter)