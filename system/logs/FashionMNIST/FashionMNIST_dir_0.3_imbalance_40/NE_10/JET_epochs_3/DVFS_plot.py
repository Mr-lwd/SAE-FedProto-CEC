import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 定义文件路径
file_path = './gr50_JET_FedSAE_DVFS_fd_64_bs_10_lr_0.06_mo_0.8_lam_1_batch_256.out'
# file_path='./DVFS_Fedproto.out'

# 初始化列表来存储提取的数据
global_rounds = []
server_global_times = []
only_train_times = []
all_energies = []


plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})
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

# New elegant color palette
color1 = '#2E86AB'  # Steel blue
color2 = '#7A9E7E'  # Sage green
color3 = '#B86B77'  # Dusty rose

LINE_WIDTH = 2.5        # 减小实线宽度
DASH_WIDTH = 3         # 虚线宽度稍大
ALPHA_SOLID = 0.7      # 实线更透明
ALPHA_DASH = 1.0       # 虚线更不透明
MARKER_INTERVAL = 5
# DASH_STYLE = (5, 3)    # 虚线样式：5个点的线段，3个点的间隔

# 绘制原始数据和拟合线
plt.plot(global_rounds, server_global_times, label='Server Global Time (Original)', 
         color=color1, alpha=ALPHA_SOLID, linewidth=LINE_WIDTH)
plt.plot(global_rounds[::MARKER_INTERVAL], server_global_time_pred[::MARKER_INTERVAL], 
         label='Server Global Time (Linear Fit)', linestyle='--', 
         color=color1, alpha=ALPHA_DASH, linewidth=DASH_WIDTH, marker='o', markersize=6)

plt.plot(global_rounds, only_train_times, label='Only Train Time (Original)', 
         color=color2, alpha=ALPHA_SOLID, linewidth=LINE_WIDTH)
plt.plot(global_rounds[::MARKER_INTERVAL], only_train_time_pred[::MARKER_INTERVAL], 
         label='Only Train Time (Linear Fit)', linestyle='--', 
         color=color2, alpha=ALPHA_DASH, linewidth=DASH_WIDTH, marker='s', markersize=6)

plt.plot(global_rounds, all_energies, label='All Energy (Original)', 
         color=color3, alpha=ALPHA_SOLID, linewidth=LINE_WIDTH)
plt.plot(global_rounds[::MARKER_INTERVAL], all_energy_pred[::MARKER_INTERVAL], 
         label='All Energy (Linear Fit)', linestyle='--',
         color=color3, alpha=ALPHA_DASH, linewidth=DASH_WIDTH, marker='^', markersize=6)

# 添加标题和标签
plt.title('Time and Energy with Linear Fit (FedSAE_DVFS)')
plt.xlabel('Iterations')
plt.ylabel('Time(s) / Energy(J)')
plt.legend()

# 显示图形
# plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("./100rounds_with_fit_and_R2_gr50_FedSAE_DVFS.png", dpi=300)
