# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients，(暂时只考虑一轮，忽略)
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

from collections import defaultdict
import copy
import random
from flcore.edges.edgebase import Edge
from flcore.clients.clientbase import load_item, save_item
from utils.func_utils import *
import torch
import numpy as np


class Edge_FedSAE(Edge):
    def __init__(self, args, id, cids):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_protos: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_protos: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param eshared_protos: SharedProtos store in edge server
        :return:
        """
        # 调用父类的构造函数
        super().__init__(args, id, cids, shared_layers=None)
        self.eshared_protos_global = None
        self.eshared_protos_local = None
        self.clients = None
        self.N_l = defaultdict(int)  # 初始化默认0
        self.N_l_prev = defaultdict(int)  # 初始化默认0
        self.eglobal_time = 0
        self.etrain_time = 0
        self.etrans_time = 0
        self.eparallel_time = 0
        self.etrans_simu_time = random.randint(10, 100)
        # Number of clients in edge l containing class j that have participated in aggregation

    def train(self, clients):

        print(f"Edge {self.id} begin training")
        selected_cnum = max(int(self.clients_per_edge * self.args.join_ratio), 1)
        self.join_clients = selected_cnum  # 记录本轮参与训练的客户端数量
        # Choose a set of clients S^l to train in parallel
        self.selected_cids = np.random.choice(
            self.cids, selected_cnum, replace=False, p=self.p_clients
        )
        for selected_cid in self.selected_cids:
            self.client_register(clients[selected_cid])

        for edge_epoch in range(self.args.edge_epochs):  # 边缘轮次, ==1
            eparallel_time_list = []
            for selected_cid in self.id_registration:
                # self.send_to_client(clients[selected_cid])
                id, train_time, trans_time = clients[selected_cid].train()
                self.etrain_time += train_time
                eparallel_time_list.append((train_time + trans_time))

            self.eparallel_time += max(eparallel_time_list)
            # self.edgeAggregate(clients)
            # self.edgeUpdate() not implement When edge_epochs is 1
        self.eglobal_time += self.eparallel_time

        self.edge_update_mean_cov(clients)

        if self.args.trans_delay_simulate is True:
            self.etrans_time += self.etrans_simu_time
            self.eglobal_time += self.etrans_time

        return self.eglobal_time, self.etrain_time, self.etrans_time

    def refresh_edgeserver(self):
        self.etrain_time = 0
        self.etrans_time = 0
        self.eparallel_time = 0
        self.receiver_buffer.clear()
        self.id_registration.clear()
        self.sample_registration.clear()

        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = client.train_samples
        return None

    def receive_from_client(self, client_id, cshared_protos_local):
        self.receiver_buffer[client_id] = cshared_protos_local
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        # sample_num = [snum for snum in self.sample_registration.values()]

    def send_to_client(self, client):
        # client.receive_from_edgeserver(copy.deepcopy(self.eshared_protos_global))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(
            edge_id=self.id,
            eshared_protos_local=copy.deepcopy(self.eshared_protos_local),
        )
        return None

    def receive_from_cloudserver(self, cloud_shared_protos=None, global_time=0):
        self.eglobal_time = global_time
        if self.args.trans_delay_simulate is True:
            self.etrans_time += self.etrans_simu_time
        # self.eshared_protos_global = cloud_shared_protos
        return None

    def edge_update_mean_cov(self, clients):
        """
        计算加权均值和协方差矩阵。
        保存边缘均值和协方差矩阵。

        Args:

        """
        clients_mean_cov = {
            client_id: load_item(
                clients[client_id].role, "mean_cov", self.save_folder_name
            )
            for client_id in self.cids
        }
        # self.N_l = defaultdict(int)
        for id in self.id_registration:
            if id not in self.have_participated_ids:
                for key in clients[id].label_counts.keys():
                    self.N_l[key] += clients[id].label_counts[key]
                self.have_participated_ids.add(id)

        self.edge_cal_mean_and_cov(clients_mean_cov)

    def edge_cal_mean_and_cov(self, clients_mean_cov):
        """
        按数据量计算全局均值和协方差矩阵。

        Args:
            clients_mean_cov (dict):
                {client_id: {label: {"mean": Tensor, "cov": Tensor, "counts": int}}}, 客户端均值和协方差，样本数量。
            N_l (dict):
                {label: total_count}, 每个类别在所有已经参加训练的客户端的样本总数。

        Returns:
            edge_mean_cov (dict):
                {label: {"mean": Tensor, "cov": Tensor}}, 全局均值和协方差。
        """
        edge_mean_cov = {}

        # Step 1: 计算边缘均值
        for label in self.N_l.keys():
            # 初始化
            edge_mean = None
            edge_count = 0
            for client_id, data in clients_mean_cov.items():
                if data is None or label not in data:
                    continue
                mean_vec = data[label]["mean"]
                client_count = data[label]["counts"]
                if edge_mean is None:
                    edge_mean = mean_vec * client_count
                else:
                    edge_mean += mean_vec * client_count

                edge_count += client_count

            edge_mean /= self.N_l[label]
            # print(f"N_l[label]:{self.N_l[label]}, client_count:{client_count}")
            edge_mean_cov[label] = {"mean": edge_mean}

        # Step 2: 计算边缘协方差
        for label in self.N_l.keys():
            total_cov = None
            for client_id, data in clients_mean_cov.items():
                if data is None or label not in data:
                    continue

                # 从客户端获取信息
                mean_vec = data[label]["mean"]
                cov_matrix = data[label]["cov"]
                client_count = data[label]["counts"]

                # 加权协方差
                # print(f"label:{label}, client_count:{client_count}, cov_matrix shape: {cov_matrix.shape}, mean_vec shape: {mean_vec.shape}, edge_mean_cov[label]['mean'] shape: {edge_mean_cov[label]['mean'].shape}")
                weighted_cov = (client_count - 1) * cov_matrix
                # print(f"weighted_cov shape: {weighted_cov.shape}")
                mean_diff = (mean_vec - edge_mean_cov[label]["mean"]).unsqueeze(
                    1
                )  # 变为列向量
                if client_count == 1:
                    weighted_cov = torch.zeros_like(
                        torch.mm(mean_diff, mean_diff.T)
                    )  # 初始化为零矩阵
                else:
                    weighted_cov += client_count * torch.mm(
                        mean_diff, mean_diff.T
                    )  # 均值差计算

                if total_cov is None:
                    total_cov = weighted_cov
                else:
                    total_cov += weighted_cov

            # 归一化
            edge_cov = total_cov / (self.N_l[label] - 1)
            edge_mean_cov[label]["cov"] = edge_cov
        save_item(
            edge_mean_cov,
            role=self.role,
            item_name="mean_cov",
            item_path=self.save_folder_name,
        )
        return edge_mean_cov
