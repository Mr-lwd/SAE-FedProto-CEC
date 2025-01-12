import matplotlib.pyplot as plt
import json
# 数据解析

datasets = ["Cifar10_dir_0.1_imbalance_40", "Cifar10_dir_0.3_imbalance_40",
            "FashionMNIST_dir_0.1_imbalance_40","FashionMNIST_dir_0.3_imbalance_40",
            "MNIST_dir_0.1_imbalance_40","MNIST_dir_0.3_imbalance_40"]

def draw_distribute(dataset_name):
    config_path = f"./{dataset_name}/config.json"
    data = json.load(open(config_path, "r"))
    
    num_clients = data["num_clients"]
    num_classes = data["num_classes"]
    size_of_samples = data["Size of samples for labels in clients"]

    # 解析数据
    x, y, sizes = [], [], []

    for client_id, labels in enumerate(size_of_samples):
        for label, count in labels:
            x.append(client_id)
            y.append(label)
            sizes.append(count)

    # 绘制图形
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        x, y, s=[size**0.7 for size in sizes], c=y, cmap='tab10', alpha=0.9, marker='o'
    )

    # 设置刻度和网格线
    plt.xticks(range(num_clients))  # 确保每个客户端都有一个刻度
    plt.yticks(range(num_classes))  # 确保每个类都有一个刻度
    plt.grid(axis='x', which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # 仅绘制竖轴网格线



    # 添加色彩条和标签
    # plt.colorbar(scatter, label="Class ID")
    plt.xlabel("Client ID")
    plt.ylabel("Class ID")
    plt.title("Data Distribution Across Clients and Classes")
    plt.savefig(f"./{dataset_name}_data_distribute.png")


for dataset in datasets:
    draw_distribute(dataset)

