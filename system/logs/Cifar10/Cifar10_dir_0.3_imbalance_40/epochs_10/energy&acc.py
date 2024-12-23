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


# 定义文件路径
log_files = [
    # "./lr_006_mo_0.5_lam_2_batch_256_FedSAE_gam_1.out"
    "./lr_008_mo_0_lam_2_batch_256_FedSAE_gam_1.out"
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
