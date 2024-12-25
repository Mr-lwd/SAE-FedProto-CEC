import pandas as pd
import matplotlib.pyplot as plt
import re

# 定义一个函数来读取日志文件并提取数据
def extract_data(log_file, time_type="all_clients_time_cost", model_type="Prototype"):
    accuracies = []
    time_costs = []

    with open(log_file, "r") as file:
        content = file.read()
        # 根据model_type选择不同的正则表达式来提取准确性
        if model_type == "Regular":
            accuracies = re.findall(
                r"Averaged Test Accuracy \(Regular Model\): (\d+\.\d+)", content
            )
        else:
            accuracies = re.findall(
                r"Averaged Test Accuracy \(Prototype Model\): (\d+\.\d+)", content
            )

        # 提取时间成本
        time_costs = re.findall(
            r"{time_type}: (\d+\.\d+)".format(time_type=time_type), content
        )

    return [float(tc) for tc in time_costs], [float(acc) for acc in accuracies]

# 定义一个函数来提取训练损失
def extract_loss_data(log_file):
    regular_losses = []
    proto_losses = []

    with open(log_file, "r") as file:
        content = file.read()
        
        # 提取 Regular Model 的训练损失
        regular_losses = re.findall(
            r"Averaged Train Loss \(Regular Model\): (\d+\.\d+)", content
        )
        
        # 提取 Regular + Proto 的训练损失
        proto_losses = re.findall(
            r"Averaged Train Loss \(Regular \+ Proto\): (\d+\.\d+)", content
        )

    return [float(loss) for loss in regular_losses], [float(loss) for loss in proto_losses]

# 定义文件路径
log_files = [
    "./FedProto_lr_006_mo_0.8_lam_1_batch_256.out",
    "./FedSAE_gam_1_lr_0.06_mo_0.8_lam_1_batch_256.out",
    "./FedTGP_lr_0.06_mo_0.8_lam_1_batch_256.out"
]

time_model_configs = [ 
    {
        "time_type": "only_train_time",
        "models": [
            ("Prototype", "./onlytrain.png"),
            ("Regular", "./onlytrain_model.png"),
        ],
    },
    {
        "time_type": "server_global_time",
        "models": [
            ("Prototype", "./global_time.png"),
            ("Regular", "./global_time_model.png"),
        ],
    },
]

# 绘制时间成本和测试准确性图表
for config in time_model_configs:
    time_type = config["time_type"]
    for model_type, save_path in config["models"]:
        plt.figure(figsize=(10, 6))

        # 从每个日志文件中提取数据
        for file in log_files:
            # 去掉 .out 后缀
            legend_label = file.rsplit(".", 1)[0]

            time_costs, accuracies = extract_data(file, time_type, model_type)
            plt.plot(time_costs, accuracies, label=legend_label)

        plt.xlabel("Time Cost")
        plt.ylabel("Averaged Test Accuracy")
        plt.title(f"Training Log Analysis ({model_type} Model)")
        plt.legend()
        plt.grid()

        # 保存图表
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")

        # 清除当前图形以便下一个循环可以重新绘图
        plt.clf()

# 绘制训练损失图表
# plt.figure(figsize=(10, 6))

# for file in log_files:
#     # 去掉 .out 后缀
#     legend_label = file.rsplit(".", 1)[0]

#     regular_losses, proto_losses = extract_loss_data(file)
    
#     # 绘制 Regular Model 的训练损失
#     plt.plot(range(1, len(regular_losses) + 1), regular_losses, label=f"{legend_label} (Regular)")
    
#     # 绘制 Regular + Proto 的训练损失
#     plt.plot(range(1, len(proto_losses) + 1), proto_losses, label=f"{legend_label} (Regular + Proto)")

# plt.xlabel("Iteration")
# plt.ylabel("Averaged Train Loss")
# plt.title("Training Loss Analysis")
# plt.legend()
# plt.grid()

# # 保存图表
# plt.savefig("./All_loss.png")
# print("Chart saved to ./All_loss.png")

# plt.clf()

# 绘制单独的 Regular 和 Regular + Proto 损失图表
plt.figure(figsize=(10, 6))

for file in log_files:
    # 去掉 .out 后缀
    legend_label = file.rsplit(".", 1)[0]

    regular_losses, _ = extract_loss_data(file)
    plt.plot(range(1, len(regular_losses) + 1), regular_losses, label=f"{legend_label}")

plt.xlabel("Iteration")
plt.ylabel("Averaged Train Loss (Regular)")
plt.title("Training Loss Analysis (Regular Model)")
plt.legend()
plt.grid()

plt.savefig("./Model_loss.png")
print("Chart saved to ./Model_loss.png")

plt.clf()
