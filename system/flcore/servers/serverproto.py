import time
import numpy as np

from flcore.clients.clientproto import clientProto
from flcore.edges.edgeproto import Edge_FedProto
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.func_utils import *
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import copy
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.glprotos_invol_dataset = defaultdict(int)
        self.global_protos_data = defaultdict(list)

        # select slow clients
        self.set_slow_clients()
        # 初始化所有客户端
        self.set_clients(clientProto)
        # 初始化所有边缘服务器
        self.set_edges(Edge_FedProto)

        self.refresh_cloudserver()

        self.compute_glprotos_invol_dataset()
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.current_epoch = 0
        self.buffer = []

    def train(self):
        for i in range(self.global_rounds + 1):  # 总论次
            s_t = time.time()
            self.selected_edges = self.select_edges()
            self.refresh_cloudserver()
            [self.edge_register(edge=edge) for edge in self.edges]
            for j, edge in enumerate(self.edges):  # 遍历所有边缘服务器
                # self.send_to_edge(edge)
                edge.train(self.clients)

            # 直接计算从id_registration读取id，读取triple
            self.cloudUpdate()

            self.Budget.append(time.time() - s_t)
            print("-" * 50, self.Budget[-1])
            #             self.all_clients_time_cost += self.Budget[-1]
            self.current_epoch += 1

            if i % self.eval_gap == 0:
                print(f"\n-------------Global Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def cloudUpdate(self):
        assert len(self.selected_edges) > 0

        self.uploaded_ids = []
        uploaded_protos = defaultdict(dict)
        for edge in self.selected_edges:
            id = edge.id
            self.uploaded_ids.append(id)
            protos = load_item(edge.role, "protos", self.save_folder_name)
            prev_protos = load_item(edge.role, "prev_protos", self.save_folder_name)
            uploaded_protos[id] = {"protos": protos, "prev_protos": prev_protos}

        global_protos = self.proto_aggregation(uploaded_protos)
        save_item(global_protos, self.role, "global_protos", self.save_folder_name)

    #    https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
    def proto_aggregation(self, edge_protos_list):
        agg_protos_label = defaultdict(default_tensor)
        global_protos = load_item(self.role, "global_protos", self.save_folder_name)
        if self.agg_type == 0:
            for j in range(self.args.num_classes):
                if global_protos is not None and j in global_protos.keys():
                    for edge in self.edges:
                        agg_protos_label[j] += edge.N_l_prev[j] * global_protos[j]
                        assert len(agg_protos_label[j]) == self.args.feature_dim
                for id in self.id_registration:
                    if (
                        edge_protos_list[id]["protos"] is not None
                        and j in edge_protos_list[id]["protos"].keys()
                    ):
                        agg_protos_label[j] += (
                            self.edges[id].N_l[j] * edge_protos_list[id]["protos"][j]
                        )
                        assert len(agg_protos_label[j]) == self.args.feature_dim
                    if (
                        edge_protos_list[id]["prev_protos"] is not None
                        and j in edge_protos_list[id]["prev_protos"].keys()
                    ):
                        agg_protos_label[j] -= (
                            self.edges[id].N_l_prev[j] * edge_protos_list[id]["prev_protos"][j]
                        )
                        assert len(agg_protos_label[j]) == self.args.feature_dim
                        
                    self.edges[id].N_l_prev[j] = self.edges[id].N_l[j]

                if agg_protos_label[j] is not None:
                    agg_protos_label[j] = agg_protos_label[j] / sum(
                        edge.N_l_prev[j] for edge in self.edges
                    )

        elif self.agg_type == 1:
            for local_protos in edge_protos_list:
                for label in local_protos["protos"].keys():
                    agg_protos_label[label].append(
                        local_protos["protos"][label]
                        * local_protos["client"].label_counts[label]
                    )

            for [label, proto_list] in agg_protos_label.items():
                if len(proto_list) > 1:
                    proto = 0 * proto_list[0].data
                    for i in proto_list:
                        proto += i.data
                    agg_protos_label[label] = proto / self.glprotos_invol_dataset[label]
                else:
                    agg_protos_label[label] = (
                        proto_list[0].data / self.glprotos_invol_dataset[label]
                    )

        print("agg_protos_label", agg_protos_label.keys())
        
        for id in self.id_registration:
            if edge_protos_list[id] is not None:
                save_item(
                    edge_protos_list[id]["protos"],
                    self.edges[id].role,
                    "prev_protos",
                    self.save_folder_name,
                )
        # if self.args.drawtsne is True and self.current_epoch % 10 == 0:
        #     save_tsne_with_agg(
        #         edge_protos_list=edge_protos_list,
        #         agg_protos_label=agg_protos_label,
        #         base_path="./tsneplot",
        #         dataset=self.args.dataset,
        #         algorithm=self.args.algorithm,
        #         local_epochs=self.args.local_epochs,
        #         agg_type=self.args.agg_type,
        #         glclassifier=self.args.glclassifier,
        #         test_useglclassifier=self.args.test_useglclassifier,
        #         gamma=self.args.gamma,
        #         lamda=self.args.lamda,
        #         lr_rate=self.args.local_learning_rate,
        #         usche=self.args.use_decay_scheduler,
        #         current_epoch=self.current_epoch,
        #     )
        return agg_protos_label

    def train_global_classifier(self):
        self.global_classifier = nn.Linear(self.feature_dim, self.num_classes)

        for param in self.global_classifier.parameters():
            param.requires_grad = True
        features = []
        labels = []
        # 生成与 features 形状相同的高斯噪声

        for label, proto_list in self.global_protos_data.items():
            features.extend(proto_list)  # 追加所有特征向量
            labels.extend([label] * len(proto_list))  # 对应的标签
        print("features.len", len(features))
        print("labels.len", len(labels))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.global_classifier.parameters(), lr=0.001)

        features = torch.stack(
            [torch.tensor(feature) for feature in features]
        )  # 将每个特征向量转换为张量，并堆叠成一个 Tensor

        labels = torch.tensor(labels)

        # 确保 features 和 labels 都在正确的设备上
        features = features.to(self.device)
        labels = labels.to(self.device)
        # 创建数据加载器
        dataset = TensorDataset(features, labels)
        data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

        # 训练过程
        epochs = 10
        print("train_global_classifier")
        for epoch in range(epochs):
            self.global_classifier.to(self.device)
            self.global_classifier.train()
            for batch_features, batch_labels in data_loader:
                # 前向传播
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.global_classifier(batch_features)
                loss = criterion(outputs, batch_labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.global_classifier.eval()  # 设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 禁用梯度计算
            # 使用验证集进行验证
            for val_features, val_labels in data_loader:
                val_features = val_features.to(self.device)
                val_labels = val_labels.to(self.device)

                # 前向传播
                val_outputs = self.global_classifier(val_features)
                _, predicted = torch.max(val_outputs, 1)  # 获取预测的类别

                # 统计正确的预测数量
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        # 计算验证集准确率
        accuracy = 100 * correct / total
        print("global classifier accuracy:", accuracy)

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        # del self.id_registration[:]
        self.id_registration.clear()
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_protos):
        self.receiver_buffer[edge_id] = eshared_protos
        return None

    def aggregate_protos(self):
        # received_dict = [dict for dict in self.receiver_buffer.values()]
        # sample_num = [snum for snum in self.sample_registration.values()]
        # self.shared_protos = average_weights(w=received_dict,
        #                                          s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_protos))
        return None

    def compute_glprotos_invol_dataset(self):
        for client in self.clients:
            for key in client.label_counts.keys():
                self.glprotos_invol_dataset[key] += client.label_counts[key]


def save_tsne_with_agg(
    edge_protos_list,
    agg_protos_label,
    base_path,
    dataset,
    algorithm,
    local_epochs,
    agg_type,
    glclassifier,
    test_useglclassifier,
    gamma,
    lamda,
    lr_rate,
    usche,
    current_epoch,
):
    """
    生成并保存包含本地和聚合原型的 t-SNE 图。
    """

    save_folder = f"{base_path}/{dataset}/{algorithm}/localepoch_{local_epochs}_agg_{agg_type}_lamda_{lamda}_glclassifier_{glclassifier}_use_{test_useglclassifier}_gamma_{gamma}_lr_{lr_rate}_usche_{usche}"

    all_features = []
    all_labels = []
    label_types = []  # 区分来源：agg 或 local

    # 收集聚合原型数据
    for label, proto in agg_protos_label.items():
        all_features.append(proto.cpu().detach().numpy())
        all_labels.append(label)
        label_types.append("Global")  # 标记为全局原型

    # 收集本地原型数据
    for local_protos in edge_protos_list:
        for label, proto in local_protos["protos"].items():
            all_features.append(proto.cpu().detach().numpy())
            all_labels.append(label)
            label_types.append("Local")  # 标记为本地原型

    # 转为 NumPy 数组
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    # print("all_features",all_features)
    # print("all_labels",all_labels)
    # 打印样本数量
    print("Number of samples:", len(all_features))

    # t-SNE 降维，调整 perplexity
    # perplexity = min(30, len(all_features) - 1)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(all_features)

    # 归一化处理
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    # reduced_features = scaler.fit_transform(reduced_features)

    # 创建保存路径
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"tsne_agg_epoch_{current_epoch}.png")

    # 可视化
    plt.figure(figsize=(10, 8))
    unique_labels = set(all_labels)
    num_labels = len(unique_labels)
    cmap = plt.get_cmap("tab10")  # 使用 Matplotlib 的内置调色盘
    colors = [cmap(i % 10) for i in range(num_labels)]  # 支持多种颜色，不重复
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        indices = all_labels == label
        label_source = np.array(label_types)[indices]

        # 分别获取全局和本地的布尔索引
        global_indices = indices.copy()
        global_indices[indices] = label_source == "Global"

        local_indices = indices.copy()
        local_indices[indices] = label_source == "Local"

        # 获取标签对应的颜色
        color = label_to_color[label]

        # 分别绘制全局和本地原型
        if np.any(global_indices):
            plt.scatter(
                reduced_features[global_indices, 0],
                reduced_features[global_indices, 1],
                label=f"Global Proto {label}",
                color=color,
                alpha=0.7,
                marker="o",
            )
        if np.any(local_indices):
            plt.scatter(
                reduced_features[local_indices, 0],
                reduced_features[local_indices, 1],
                label=f"Local Proto {label}",
                color=color,
                alpha=0.5,
                marker="x",
            )
    # 添加图例
    plt.legend()
    plt.title(
        f"t-SNE Visualization with Global and Local Prototypes (Epoch {current_epoch})"
    )
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(save_path)
    plt.close()
