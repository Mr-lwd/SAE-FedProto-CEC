import matplotlib.pyplot as plt

# 数据
Lamdas = [0.1, 0.5, 1, 2, 5]
FedProto_model_acc = [66.78,66.47,66.97,66.82,66.02]
FedProto_prototype_acc = [66.86,66.45,66.95,66.81,66.10]
Loss=[]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制 FedProto_model_acc 曲线，点加粗
plt.plot(Lamdas, FedProto_model_acc, marker='o', markersize=8, label='FedSAE Model Accuracy', linewidth=2)

# 绘制 FedProto_prototype_acc 曲线，点加粗
plt.plot(Lamdas, FedProto_prototype_acc, marker='s', markersize=8, label='FedSAE Prototype Accuracy', linewidth=2)

# 添加标题和标签SAE
plt.title('Accuracy vs. Lamdas (FedSAE)', fontsize=16)
plt.xlabel('λ', fontsize=14)
plt.ylabel('Accuracy(%)', fontsize=14)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(fontsize=12)

# 显示图形
plt.show()

# 保存图像，dpi=300，bbox_inches='tight'
plt.savefig('accuracy_vs_lamdas_FedSAE.png', dpi=300, bbox_inches='tight')
