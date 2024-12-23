import time
import numpy as np
from flcore.clients.clienttgp import clientTGP
from flcore.edges.edgetgp import Edge_FedTGP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.func_utils import *
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import copy
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.glprotos_invol_dataset = defaultdict(int)

        # select slow clients
        self.set_slow_clients()
        # 初始化所有客户端
        self.set_clients(clientTGP)
        # 初始化所有边缘服务器
        self.set_edges(Edge_FedTGP)

        self.compute_glprotos_invol_dataset()
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.current_epoch = 0
        self.buffer = []
        self.global_time = 0
        self.gtrain_time = 0
        self.gtrans_time = 0
        self.readyList = DynamicBuffer(self.num_edges)
        self.tobetrained = DynamicBuffer(self.num_edges)
        self.aggregation_buffer = DynamicBuffer(self.num_edges)
        [self.edge_register(edge=edge) for edge in self.edges]

        self.server_learning_rate = args.server_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        self.TGP_uploaded_protos = []
        if args.save_folder_name == "temp" or "temp" not in args.save_folder_name:
            PROTO = Trainable_prototypes(
                self.num_classes, self.server_hidden_dim, self.feature_dim, self.device
            ).to(self.device)
            save_item(PROTO, self.role, "PROTO", self.save_folder_name)
            print(PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

    def train(self):
        for i in range(self.global_rounds + 1):  # 总论次
            # s_t = time.time()
            self.selected_edges = self.select_edges()
            self.refresh_cloudserver()
            # [self.edge_register(edge=edge) for edge in self.edges]
            print("tobetrained:")
            self.tobetrained.printTimeinfo()
            for edge in self.tobetrained.buffer:
                edge.refresh_edgeserver()
                edge.receive_from_cloudserver(global_time=self.global_time)
                eglobal_time, etrain_time, etrans_time = edge.train(self.clients)
                self.gtrain_time += etrain_time
                endTrainEdge = self.tobetrained.getbyid(edge.id)
                self.readyList.add(endTrainEdge)
            self.tobetrained.buffer = []
            print("readyList:")
            self.readyList.printTimeinfo()
            self.trans_aggedges_from_readyList()
            self.global_time = self.aggregation_buffer.buffer[-1].eglobal_time
            # 直接计算从id_registration读取id，读取triple
            self.cloudUpdate()
            print("end Update")
            self.push_aggclients_to_trainList()

            print("server_global_time:", self.global_time)
            print("only_train_time:", self.gtrain_time)
            # self.Budget.append(time.time() - s_t)
            # print("-" * 50, self.Budget[-1])
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
        # print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def cloudUpdate(self):
        assert len(self.selected_edges) > 0
        assert len(self.aggregation_buffer.buffer) > 0
        print("aggregation_buffer:")
        self.aggregation_buffer.printTimeinfo()

        self.tgp_process()

        self.save_tsne_with_agg(
            args=self.args,
            base_path="./tsneplot",
            drawtype="clientavgproto",
            current_epoch=self.current_epoch,
        )

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        # del self.id_registration[:]
        self.id_registration.clear()
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.tobetrained.add(edge)
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

    def trans_aggedges_from_readyList(self):
        while (
            len(self.aggregation_buffer.buffer) < self.aggregation_buffer.max_length
            and len(self.readyList.buffer) > 0
        ):
            # for step in range(self.async_buffer_length):
            self.aggregation_buffer.add(self.readyList.buffer.pop(0))

    def push_aggclients_to_trainList(self):
        while len(self.aggregation_buffer.buffer) > 0:
            self.tobetrained.add(self.aggregation_buffer.buffer.pop(0))

    def tgp_process(self):
        uploaded_protos_per_client = []
        self.TGP_uploaded_protos = []
        for client in self.clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, "protos", client.save_folder_name)
            uploaded_protos_per_client.append({"client": client, "protos": protos})
            for k in protos.keys():
                self.TGP_uploaded_protos.append((protos[k], k))

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        global_protos = self.proto_cluster(uploaded_protos_per_client)
        # global_protos = self.proto_aggregation(uploaded_protos_per_client)
        # save_item(global_protos, self.role, "global_protos", self.save_folder_name)

        for k1 in global_protos.keys():
            for k2 in global_protos.keys():
                if k1 > k2:
                    dis = torch.norm(global_protos[k1] - global_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)

        self.update_Gen()

    def update_Gen(self):
        PROTO = load_item(self.role, "PROTO", self.save_folder_name)
        Gen_opt = torch.optim.Adam(PROTO.parameters(), lr=0.001)
        # Gen_opt = torch.optim.SGD(PROTO.parameters(), lr=self.server_learning_rate)
        PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(
                self.TGP_uploaded_protos, self.batch_size, drop_last=False, shuffle=True
            )
            for proto, y in proto_loader:
                y = torch.tensor(y).to(self.device, dtype=torch.int64)  # 转换为整数类型
                # y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = PROTO(list(range(self.num_classes)))
                proto = proto.squeeze(1)  # 移除第二维，proto 变为 [24, 512]

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                # exit()
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                margin = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * margin
                loss = self.CEloss(-dist, y)
                ##添加类内约束
                if self.args.tgpaddmse == 1:
                    # 使用索引操作从 proto_gen 中获取对应类中心
                    selected_centers = proto_gen[y]  # [batch_size, feature_dim]
                    # 计算类内约束损失
                    loss += self.MSEloss(selected_centers, proto) * self.args.lamda

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        print(f"Server loss: {loss.item()}")
        self.TGP_uploaded_protos = []
        save_item(PROTO, self.role, "PROTO", self.save_folder_name)

        PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = PROTO(
                torch.tensor(class_id, device=self.device)
            ).detach()
        save_item(global_protos, self.role, "global_protos", self.save_folder_name)

        # agg_protos_label = defaultdict(list)
        # for local_protos in protos_list:
        #     for label in local_protos["protos"].keys():
        #         agg_protos_label[label].append(
        #             local_protos["protos"][label]
        #             * local_protos["client"].label_counts[label]
        #         )

        # for [label, proto_list] in agg_protos_label.items():
        #     if len(proto_list) > 1:
        #         proto = 0 * proto_list[0].data
        #         for i in proto_list:
        #             proto += i.data
        #         agg_protos_label[label] = proto / self.glprotos_invol_dataset[label]
        #     else:
        #         agg_protos_label[label] = (
        #             proto_list[0].data / self.glprotos_invol_dataset[label]
        #         )
        # return agg_protos_label
