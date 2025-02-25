import matplotlib.pyplot as plt

# 数据
vfns = [500, 1000, 2000, 4000, 6000, 10000]
FedSAE_model_acc = [66.30, 66.11, 66.39, 67.22, 66.31, 66.41]
FedSAE_prototype_acc = [66.33, 66.23, 66.40, 67.27, 66.28, 66.45]
Loss=[]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制 FedSAE_model_acc 曲线，点加粗
plt.plot(vfns, FedSAE_model_acc, marker='o', markersize=8, label='FedSAE Model Accuracy', linewidth=2)

# 绘制 FedSAE_prototype_acc 曲线，点加粗
plt.plot(vfns, FedSAE_prototype_acc, marker='s', markersize=8, label='FedSAE Prototype Accuracy', linewidth=2)

# 添加标题和标签SAEe

plt.title('Accuracy vs. vfns (FedSAE)', fontsize=16)
plt.xlabel('The number of virtual features generated for each class', fontsize=14)
plt.ylabel('Accuracy(%)', fontsize=14)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(fontsize=12)

# 显示图形
plt.show()

# 保存图像，dpi=300，bbox_inches='tight'
plt.savefig('accuracy_vs_vfns_FedSAE.png', dpi=200, bbox_inches='tight')
