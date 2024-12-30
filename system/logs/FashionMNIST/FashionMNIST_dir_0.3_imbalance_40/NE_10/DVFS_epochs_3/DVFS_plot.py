import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 定义文件路径
file_path = './FedProto_DVFS_1_lr_0.06_mo_0.8_lam_1_batch_256.out'

# 初始化列表来存储提取的数据
global_rounds = []
server_global_times = []
only_train_times = []
all_energies = []

# 打开文件并读取
with open(file_path, 'r') as file:
    content = file.read()

    # 使用正则表达式提取所需的数据
    global_rounds = re.findall(r'-------------Global Round number: (\d+)', content)
    server_global_times = re.findall(r'server_global_time: ([\d\.]+)', content)
    only_train_times = re.findall(r'only_train_time: ([\d\.]+)', content)
    all_energies = re.findall(r'All Energy: ([\d\.]+)', content)

# 转换为浮点数
global_rounds = [int(round_num) for round_num in global_rounds]
server_global_times = [float(time) for time in server_global_times]
only_train_times = [float(time) for time in only_train_times]
all_energies = [float(energy) for energy in all_energies]

# 计算线性回归拟合
def fit_linear_model(x, y):
    # 将数据转化为NumPy数组并reshape为列向量
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    
    model = LinearRegression()
    model.fit(x, y)
    
    # 预测值
    y_pred = model.predict(x)
    
    # 计算均方误差（MSE）
    mse = mean_squared_error(y, y_pred)
    
    # 计算 R^2
    r2 = r2_score(y, y_pred)
    
    return model, y_pred, mse, r2

# 对每个变量进行线性拟合并计算偏差和R^2
server_global_time_model, server_global_time_pred, server_global_time_mse, server_global_time_r2 = fit_linear_model(global_rounds, server_global_times)
only_train_time_model, only_train_time_pred, only_train_time_mse, only_train_time_r2 = fit_linear_model(global_rounds, only_train_times)
all_energy_model, all_energy_pred, all_energy_mse, all_energy_r2 = fit_linear_model(global_rounds, all_energies)

# 打印拟合模型和均方误差（偏差）及 R^2
print(f"Server Global Time Linear Model: y = {server_global_time_model.coef_[0]:.4f}x + {server_global_time_model.intercept_:.4f}")
print(f"Server Global Time MSE (偏差): {server_global_time_mse:.4f}")
print(f"Server Global Time R^2: {server_global_time_r2:.4f}")

print(f"Only Train Time Linear Model: y = {only_train_time_model.coef_[0]:.4f}x + {only_train_time_model.intercept_:.4f}")
print(f"Only Train Time MSE (偏差): {only_train_time_mse:.4f}")
print(f"Only Train Time R^2: {only_train_time_r2:.4f}")

print(f"All Energy Linear Model: y = {all_energy_model.coef_[0]:.4f}x + {all_energy_model.intercept_:.4f}")
print(f"All Energy MSE (偏差): {all_energy_mse:.4f}")
print(f"All Energy R^2: {all_energy_r2:.4f}")

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制原始数据和拟合线
plt.plot(global_rounds, server_global_times, label='Server Global Time (Original)', color='blue')
plt.plot(global_rounds, server_global_time_pred, label='Server Global Time (Linear Fit)', linestyle='--', color='blue')

plt.plot(global_rounds, only_train_times, label='Only Train Time (Original)', color='green')
plt.plot(global_rounds, only_train_time_pred, label='Only Train Time (Linear Fit)', linestyle='--', color='green')

plt.plot(global_rounds, all_energies, label='All Energy (Original)', color='red')
plt.plot(global_rounds, all_energy_pred, label='All Energy (Linear Fit)', linestyle='--', color='red')

# 添加标题和标签
plt.title('Global Round vs Time and Energy with Linear Fit')
plt.xlabel('Global Round Number')
plt.ylabel('Values')
plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("./DVFS_with_fit_and_R2.png", dpi=300)
