import time
import numpy as np
from flcore.clients.clientSAE_DVFS import clientSAE_DVFS
from flcore.edges.edgeSAE_DVFS import Edge_FedSAE_DVFS
from flcore.servers.serverbase import Server
# from flcore.clients.clientbase import load_item, save_item
from utils.io_utils import load_item, save_item
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
import torch.nn.functional as F
import ctypes 

class FedSAE_DVFS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.glprotos_invol_dataset = defaultdict(int)

        # select slow clients
        self.set_slow_clients()
        # 初始化所有客户端
        self.set_clients(clientSAE_DVFS)
        # 初始化所有边缘服务器
        self.set_edges(Edge_FedSAE_DVFS)

        self.compute_glprotos_invol_dataset()
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.current_epoch = 0
        self.buffer = []
        self.global_time = 0
        self.gtrain_time = 0
        self.gtrans_time = 0
        self.N = defaultdict(int)
        self.have_participated = set()
        self.readyList = DynamicBuffer(self.num_edges)
        self.tobetrained = DynamicBuffer(self.num_edges)
        self.aggregation_buffer = DynamicBuffer(self.buffersize)
        [self.edge_register(edge=edge) for edge in self.edges]
        self.global_classifier_init = nn.Linear(self.feature_dim, self.num_classes)
        self.global_classifier = copy.deepcopy(self.global_classifier_init)
        
        if self.args.jetson == 1:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, 'change_config_noprint.so')
            # 加载共享库
            print("lib_path", lib_path)
            self.cLib = ctypes.CDLL(lib_path)
        
        # save(item=self.global_classifier, role=self.role, item_name="glclassifier", item_path=self.save_folder_name)

        self.server_learning_rate = args.server_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        self.TGP_uploaded_protos = []
        # if args.save_folder_name == "temp" or "temp" not in args.save_folder_name:
        #     PROTO = Trainable_prototypes(
        #         self.num_classes, self.server_hidden_dim, self.feature_dim, self.device
        #     ).to(self.device)
        #     save_item(PROTO, self.role, "PROTO", self.save_folder_name)
        #     print(PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None
        if self.args.jetson == 1:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, 'change_config_noprint.so')
            # 加载共享库
            print("lib_path", lib_path)
            self.cLib = ctypes.CDLL(lib_path)

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
            
            time.sleep(1)
            self.cLib.changeCpuFreq(self.maxCPUfreq)
            time.sleep(1)
            
            self.cloudUpdate()
            print("end Update")
            self.push_aggclients_to_trainList()

            print("server_global_time:", self.global_time)
            print("only_train_time:", self.gtrain_time)
            # self.Budget.append(time.time() - s_t)
            # print("-" * 50, self.Budget[-1])
            #             self.all_clients_time_cost += self.Budget[-1]
            self.current_epoch += 1

            if i % self.eval_gap == 0 or self.global_rounds - i < 4:
                print(f"\n-------------Global Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate_proto()

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
        self.uploaded_ids = []
        self.uploaded_client_ids = []
        for edge in self.aggregation_buffer.buffer:
            id = edge.id
            self.uploaded_ids.append(id)
            for client_id in edge.id_registration:
                self.uploaded_client_ids.append(client_id)
                clientprotos = load_item(
                    self.clients[client_id].role, "protos", self.save_folder_name
                )
                save_item(
                    clientprotos,
                    self.clients[client_id].role,
                    "cloud_protos",
                    self.save_folder_name,
                )
        # global_protos = self.proto_aggregation_clients()

        sampled_features = self.cal_meancov_and_saveglprotos()
        if self.args.drawGMM == 1 and self.current_epoch == 1:
            origin_features = defaultdict(list)
            # Select a class for visualization
            label_to_vis = 0
            for client in self.clients:
                client_features = load_item(client.role, "features", client.save_folder_name)
                features = client_features[label_to_vis]
                # Handle CUDA tensors properly
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                elif isinstance(features, list):
                    # If it's a list of tensors, convert each tensor
                    features = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in features]
                origin_features[label_to_vis].extend(features)
            
            # Prepare data
            sampled_data = np.array(sampled_features[label_to_vis])
            origin_data = np.array(origin_features[label_to_vis])
            
            # Fit Gaussian distribution to original features
            mean = np.mean(origin_data, axis=0)
            cov = np.cov(origin_data.T)
            gaussian_samples = np.random.multivariate_normal(mean, cov, size=4000)
            
            # Plot original vs Gaussian samples
            tsne = TSNE(n_components=2, random_state=42)
            combined_data = np.vstack([gaussian_samples, origin_data])
            labels = np.array(['gaussian samples'] * len(gaussian_samples) + ['original features'] * len(origin_data))
            reduced_data = tsne.fit_transform(combined_data)
            
            plt.figure(figsize=(10, 8))
            colors = {'gaussian samples': '#87CEEB', 'original features': '#ff7f0e'}
            for label, marker, alpha in [('gaussian samples', 'o', 0.8), ('original features', '^', 0.8)]:
                mask = labels == label
                plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                          label=label, marker=marker, alpha=alpha, color=colors[label],
                          edgecolor='white', linewidth=0.5)

            plt.title(f't-SNE: Original vs Gaussian Generated Features (Class {label_to_vis})', fontsize=12)
            plt.legend(frameon=True, framealpha=0.8)
            plt.grid(False)
            plt.savefig(f'./gaussian_vs_original_class_{label_to_vis}.png', bbox_inches='tight', dpi=300)
            plt.close()

            # Plot original vs sampled features
            combined_data = np.vstack([sampled_data, origin_data])
            labels = np.array(['virtual features'] * len(sampled_data) + ['original features'] * len(origin_data))
            reduced_data = tsne.fit_transform(combined_data)
            
            plt.figure(figsize=(10, 8))
            colors = {'virtual features': '#0000FF', 'original features': '#ff7f0e'}
            for label, marker, alpha in [('virtual features', 'o', 0.8), ('original features', '^', 0.8)]:
                mask = labels == label
                plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                          label=label, marker=marker, alpha=alpha, color=colors[label],
                          edgecolor='white', linewidth=0.5)

            plt.title(f't-SNE: Virtual vs Original Features (Class {label_to_vis})', fontsize=12)
            plt.legend(frameon=True, framealpha=0.8)
            plt.grid(False)
            plt.savefig(f'./tsne_class_{label_to_vis}.png', bbox_inches='tight', dpi=300)
            plt.close()
            exit()

        # sampler = GaussianSampler(self.args)
        # sampled_features = sampler.aggregate_and_sample(self.edges, self.clients)
        self.train_global_classifier(sampled_features)

        # save_item(global_protos, self.role, "global_protos", self.save_folder_name)
        # if self.args.addTGP == 1:
        #     self.tgp_process()

        self.save_tsne_with_agg(
            args=self.args,
            base_path="./tsneplot",
            drawtype="clientavgproto",
            current_epoch=self.current_epoch,
        )
        
        # self.save_tsne_with_agg(
        #     args=self.args,
        #     base_path="./tsneplot",
        #     drawtype="clientallfeatures",
        #     current_epoch=self.current_epoch,
        # )

    def train_global_classifier(self, retrain_vr):
        if self.args.jetson == 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = self.device
        
        glclassifier_time_start = time.perf_counter()
        self.global_classifier = copy.deepcopy(self.global_classifier_init)
        self.global_classifier = self.global_classifier.to(device)
        
        for param in self.global_classifier.parameters():
            param.requires_grad = True
        features = []
        labels = []
        for label, proto_list in retrain_vr.items():
            features.extend(proto_list)  # 追加所有特征向量
            labels.extend([label] * len(proto_list))  # 对应的标签
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.global_classifier.parameters(), lr=0.001)

        features = torch.stack(
            [torch.tensor(feature, device=device) for feature in features]
        )  # 将每个特征向量转换为张量，并堆叠成一个 Tensor

        labels = torch.tensor(labels, device=device)

        # 确保 features 和 labels 都在正确的设备上
        # features = features.to(device)
        # labels = labels.to(device)
        # 创建数据加载器
        dataset = TensorDataset(features, labels)
        data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

        # 训练过程
        epochs = 10
        print("train global classifier")
        self.global_classifier.train()
        for epoch in range(epochs):
            for batch_features, batch_labels in data_loader:
                # 前向传播
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = self.global_classifier(batch_features)
                loss = criterion(outputs, batch_labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        glclassifier_time_end = time.perf_counter()
        print(
            "train glclassifier time:",
            glclassifier_time_end - glclassifier_time_start,
        )
        # 保存分类器
        save_item(
            self.global_classifier, self.role, "glclassifier", self.save_folder_name
        )
        self.global_classifier.eval()  # 设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 禁用梯度计算
            # 使用训练集进行验证
            for val_features, val_labels in data_loader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)

                # 前向传播
                val_outputs = self.global_classifier(val_features)
                _, predicted = torch.max(val_outputs, 1)  # 获取预测的类别

                # 统计正确的预测数量
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        # 计算验证集准确率
        accuracy = 100 * correct / total
        print("global classifier accuracy in train virtual:", accuracy)

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

    def cal_meancov_and_saveglprotos(self):
        """
        cloud_mean_cov (dict):
            {label: {"mean": Tensor, "cov": Tensor}}, 均值和协方差
        """
        self.cloud_cal_mean_cov()
        cloud_mean_cov = load_item(self.role, "mean_cov", self.save_folder_name)
        sampled_features = defaultdict(list)
        for label, item in cloud_mean_cov.items():
            if item is not None:
                # print("item[mean]", item["mean"])
                # print("item[cov]", item["cov"])
                sampled_features[label] = self._gaussian_sampling(
                    item["mean"].cpu().numpy(), item["cov"].cpu().numpy(), 4000
                )
        return sampled_features

    def cloud_cal_mean_cov(self):
        """
        计算加权均值和协方差矩阵。
        保存云端均值和协方差矩阵。

        Args:

        """
        for id in self.uploaded_ids:
            edge_mean_cov = load_item(
                self.edges[id].role, "mean_cov", self.save_folder_name
            )
            save_item(
                edge_mean_cov,
                self.edges[id].role,
                "cloud_mean_cov",
                self.save_folder_name,
            )

            if id not in self.have_participated:
                for key in self.edges[id].N_l.keys():
                    self.N[key] += self.edges[id].N_l[key]
                self.have_participated.add(id)
        # print(f"self.N[key]:{self.N[key]}")

        cloud_mean_cov = {}
        edges_mean_cov = {
            edge.id: load_item(edge.role, "cloud_mean_cov", edge.save_folder_name)
            for edge in self.edges
        }
        # Step 1: 计算边缘均值
        for label in self.N.keys():
            # 初始化
            cloud_mean = None
            cloud_count = 0
            for edge_id, data in edges_mean_cov.items():
                if data is None or label not in data:
                    continue
                mean_vec = data[label]["mean"]
                edge_count = self.edges[edge_id].N_l[label]
                if cloud_mean is None:
                    cloud_mean = mean_vec * edge_count
                else:
                    cloud_mean += mean_vec * edge_count

                cloud_count += edge_count

            cloud_mean /= self.N[label]
            # print(f"N[label]:{self.N[label]}, cloud_count:{cloud_count}")
            cloud_mean_cov[label] = {"mean": cloud_mean}

        # Step 2: 计算边缘协方差
        for label in self.N.keys():
            total_cov = None
            for edge_id, data in edges_mean_cov.items():
                if data is None or label not in data:
                    continue

                # 从客户端获取信息
                mean_vec = data[label]["mean"]
                cov_matrix = data[label]["cov"]

                # 加权协方差
                weighted_cov = (self.edges[edge_id].N_l[label] - 1) * cov_matrix
                mean_diff = (mean_vec - cloud_mean_cov[label]["mean"]).unsqueeze(
                    1
                )  # 变为列向量
                if self.edges[edge_id].N_l[label] == 1:
                    weighted_cov = torch.zeros_like(
                        torch.mm(mean_diff, mean_diff.T)
                    )  # 初始化为零矩阵
                else:
                    weighted_cov += self.edges[edge_id].N_l[label] * torch.mm(
                        mean_diff, mean_diff.T
                    )
                if total_cov is None:
                    total_cov = weighted_cov
                else:
                    total_cov += weighted_cov

            # 归一化
            edge_cov = total_cov / (self.N[label] - 1)
            cloud_mean_cov[label]["cov"] = edge_cov
        save_item(
            cloud_mean_cov,
            self.role,
            "mean_cov",
            self.save_folder_name,
        )

        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            if class_id in cloud_mean_cov:
                global_protos[class_id] = cloud_mean_cov[class_id]["mean"]
        global_protos = {
            j: pro.to(self.device) if isinstance(pro, torch.Tensor) else torch.tensor(pro).to(self.device)
            for j, pro in global_protos.items()
        }
        save_item(global_protos, self.role, "global_protos", self.save_folder_name)

        return cloud_mean_cov

    def _gaussian_sampling(self, mean, cov, num_samples):
        """
        根据均值和协方差矩阵进行高斯采样。
        :param mean: 均值向量。
        :param cov: 协方差矩阵。
        :param num_samples: 采样数量。
        :return: 采样结果，形状为 (num_samples, feature_dim)。
        """
        sampled = np.random.multivariate_normal(mean, cov, num_samples)
        return torch.tensor(sampled, dtype=torch.float32)

    def tgp_process(self):
        self.TGP_uploaded_protos = []
        for client in self.clients:
            clientprotos = load_item(
                client.role, "cloud_protos", client.save_folder_name
            )
            if clientprotos is not None:
                for k in clientprotos.keys():
                    self.TGP_uploaded_protos.append((clientprotos[k], k))

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        global_protos = load_item(self.role, "global_protos", self.save_folder_name)
        # global_protos = self.proto_cluster(uploaded_protos_per_client)

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
        print("class-wise minimum distance", self.gap)
        print("min_gap", self.min_gap)
        print("max_gap", self.max_gap)

        self.update_Gen()

    def update_Gen(self):
        PROTO = load_item(self.role, "PROTO", self.save_folder_name)
        Gen_opt = torch.optim.SGD(PROTO.parameters(), lr=self.server_learning_rate)
        PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(
                self.TGP_uploaded_protos, self.batch_size, drop_last=False, shuffle=True
            )
            for proto, y in proto_loader:
                proto = proto.squeeze(1)  # 移除多余维，proto 变为 [24, 512]
                y = torch.tensor(y).to(self.device, dtype=torch.int64)  # 转换为整数类型
                # y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = PROTO(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                margin = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * margin
                loss = self.CEloss(-dist, y)

                ##添加类内约束
                if self.args.tgpaddmse == 1:
                    proto_new = copy.deepcopy(proto.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(proto_gen[y_c]) != type([]):
                            proto_new[i, :] = proto_gen[y_c].data
                    loss += self.MSEloss(proto_new, proto) * self.lamda

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
        save_item(global_protos, self.role, "tgp_global_protos", self.save_folder_name)
