import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from flcore.clients.clientbase import load_item, save_item


def default_tensor(feature_dim=512):
    return torch.zeros(feature_dim)


def get_transmission_time(M=0):
    # 定义所有的参数, hierachical论文设置
    B = 1 * 10**6  # 信道带宽，1 MHz = 10^6 Hz
    h = 10**-8  # 信道增益
    p = 0.5  # 发射功率，0.5 W
    sigma = 10**-10  # 噪声功率，10^-10 W
    division = 51  # 1 + (h * p) / sigma = 51
    # 计算通信延迟
    T_comm = M / (B * math.log2(division))  # 计算通信延迟
    E_comm = p * T_comm  # 计算通信能耗

    return T_comm, E_comm


def get_theory_bytes(protos={}):
    all_bytes = 0
    for key, tensor_list in protos.items():
        if isinstance(tensor_list, torch.Tensor):
            tensor_size_bytes = tensor_list.element_size() * tensor_list.nelement()
            all_bytes += tensor_size_bytes

        elif hasattr(tensor_list, "__iter__"):
            for tensor in tensor_list:
                if isinstance(tensor, torch.Tensor):
                    tensor_size_bytes = tensor.element_size() * tensor.nelement()
                    all_bytes += tensor_size_bytes
        else:
            print(f"Key {key}: tensor_list is neither a tensor nor a list.")

    return all_bytes


class DynamicBuffer:
    def __init__(self, max_length):
        self.buffer = []  # 初始化空缓冲区
        self.max_length = max_length  # 设置最大长度

    def add(self, edge):
        # 检查edge_id是否已存在
        if any(existing_edge.id == edge.id for existing_edge in self.buffer):
            print(f"edge with ID {edge.edge_id} already exists in the buffer.")
            exit(0)

        if len(self.buffer) < self.max_length:
            # 使用插入排序的方式插入
            index = 0
            while (
                index < len(self.buffer)
                and self.buffer[index].eglobal_time < edge.eglobal_time
            ):
                index += 1
            self.buffer.insert(index, edge)  # 按照global_time插入
        else:
            print("Buffer is full, cannot add more objects.")  # 缓冲区已满

    def getbyid(self, edge_id):
        for index, edge in enumerate(self.buffer):
            if edge.id == edge_id:  # 匹配edge_id
                return self.buffer[index]  # 返回匹配的对象
        print("Edge with ID {} not found.".format(edge_id))  # 未找到该edge_id

    def removebyid(self, edge_id):
        for index, edge in enumerate(self.buffer):
            if edge.id == edge_id:  # 匹配edge_id
                return self.buffer.pop(index)  # 移除匹配的对象
        print("Edge with ID {} not found.".format(edge_id))  # 未找到该edge_id

    def get_buffer(self):
        return self.buffer

    def transfer_edge_from(self, other_buffer):
        if other_buffer.buffer:  # 检查另一个缓冲区是否为空
            obj = other_buffer.remove()  # 从另一个缓冲区移除对象
            self.add(obj)  # 尝试添加到当前缓冲区

    def printTimeinfo(self):
        print([edge.id for edge in self.buffer])


class GaussianSampler:
    def __init__(self, args):
        self.args = args

    def aggregate_and_sample(self,edges, clients):
        """
        对于每个类的原型向量，根据数据量加权计算均值和协方差矩阵，并进行高斯采样。
        :param gl_all_protos: 所有的边缘服务器原型，格式为 [[features,weight],[features,weight]]。
        :return: 采样后的特征。
        """
        gl_all_protos = defaultdict(list)
        sampled_features = defaultdict(list)
        if self.args.gl_use_clients != 1:
            for edge in edges:
                edge_protos = load_item(edge.role, "prev_protos", edge.save_folder_name)
                if edge_protos is not None:
                    for key in edge_protos.keys():
                        gl_all_protos[key].append(
                            [edge_protos[key], edge.N_l_prev[key]]
                        )
        elif self.args.gl_use_clients == 1:
            for client in clients:
                client_protos = load_item(
                    client.role, "cloud_protos", client.save_folder_name
                )
                if client_protos is not None:
                    for key in client_protos.keys():
                        gl_all_protos[key].append(
                            [client_protos[key], client.label_counts[key]]
                        )

        for key in range(self.args.num_classes):
            if key in gl_all_protos.keys() and len(gl_all_protos[key]) > 0:
                # 提取每个客户端提供的原型和对应的数据量
                protos = gl_all_protos[key]  # List of (proto, client_data_size)
                weights = np.array([data[1] for data in protos], dtype=np.float32)
                features = [data[0] for data in protos]

                # 计算加权均值和协方差
                mean, cov = self._cal_weighted_mean_cov(features, weights)

                # 进行高斯采样
                num_samples = 4000
                sampled_features[key] = self._gaussian_sampling(mean, cov, num_samples)
                sampled_features[key] = torch.tensor(
                    sampled_features[key], dtype=torch.float32
                )
        return sampled_features

    def _cal_weighted_mean_cov(self, features, weights):
        """
        计算加权均值和协方差矩阵。
        :param features: 特征列表，形状为 (n, feature_dim)。
        :param weights: 权重列表，形状为 (n,)。
        :return: 加权均值和协方差矩阵。
        """
        features = torch.stack(features).cpu().numpy()  # 转为 NumPy 数组
        # 按权重归一化
        normalized_weights = weights / weights.sum()
        mean = np.average(features, axis=0, weights=normalized_weights)  # 加权均值

        # 加权协方差矩阵
        n_c = np.sum(weights)  # 类别c的总样本量
        cov = np.zeros((features.shape[1], features.shape[1]))

        for i in range(len(features)):
            mean_diff = features[i] - mean  # 样本均值与全局均值的差
            cov += weights[i] * np.outer(mean_diff, mean_diff)  # 加权外积

        # 归一化
        assert n_c > 1
        cov /= (n_c - 1)  # 除以 (n_c - 1),无偏，可能出现除0

        return mean, cov

    def _gaussian_sampling(self, mean, cov, num_samples):
        """
        根据均值和协方差矩阵进行高斯采样。
        :param mean: 均值向量。
        :param cov: 协方差矩阵。
        :param num_samples: 采样数量。
        :return: 采样结果，形状为 (num_samples, feature_dim)。
        """
        sampled = np.random.multivariate_normal(mean, cov, num_samples)
        return torch.tensor(sampled)


class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(nn.Linear(feature_dim, server_hidden_dim), nn.ReLU())]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out
