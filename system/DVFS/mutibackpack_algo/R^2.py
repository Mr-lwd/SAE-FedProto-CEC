import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json

with open('frequency&power.json', 'r') as file:
    data = json.load(file)

# 提取frequency和average_power字段
frequencies = [ item["frequency"] for item in data]
power = [ item["average_power"] for item in data]

# 将频率和功耗转换为numpy数组
frequencies = np.array(frequencies)
power = np.array(power)

# 将频率升至三次方并进行单位转换
frequencies = frequencies.astype(float)
frequencies /= 1e6  # 将频率单位转换为MHz

# 将功耗转换为千瓦 (除以1000)
power /= 1e3

# 计算频率的三次方
frequencies_cubed = frequencies ** 3

# 使用线性回归拟合
model = LinearRegression()
model.fit(frequencies_cubed.reshape(-1, 1), power)

# 获取拟合结果
k = model.coef_[0]  # 比例系数
P_static = model.intercept_  # 静态功耗
r_squared = model.score(frequencies_cubed.reshape(-1, 1), power)  # R^2

# 打印结果
print(f"k: {k}")
print(f"P_static: {P_static}")
print(f"R^2: {r_squared}")

# 使用拟合模型计算预测的功耗
predicted_power = model.predict(frequencies_cubed.reshape(-1, 1))

# 绘制原始数据和拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(frequencies, power, color='green', alpha=0.6,label='True Power')
plt.plot(frequencies, predicted_power, color='orange', alpha=0.6, label='Fit Power', linewidth=2)

# 设置标题和标签
plt.title('Relationship between Frequency and Power')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Power (W)')
plt.legend()

# 显示图形
plt.show()
plt.savefig('FittingPower.png', dpi=300, bbox_inches='tight')
