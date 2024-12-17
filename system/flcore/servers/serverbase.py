from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item
from utils.func_utils import *
from collections import defaultdict
import torch.nn as nn


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.edge_epochs = args.edge_epochs
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.num_edges = args.num_edges
        self.clients_per_edge = int(args.num_clients / args.num_edges)
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.role = "Server"
        if args.save_folder_name == "temp":
            args.save_folder_name_full = f"{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/"
        elif "temp" in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = (
                f"{args.save_folder_name}/{args.dataset}/{args.algorithm}/"
            )
        self.save_folder_name = args.save_folder_name_full

        self.selected_edges = []

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.edges = []
        self.p_edge = []  # 每个服务器上的边缘服务器数据量比例
        self.id_registration = []
        self.sample_registration = {}
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.clock = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_client_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.first_client_epoch_times = []
        self.parallel_time_cost = 0
        self.only_train_time = 0
        self.feature_dim = args.feature_dim

        self.total_counts = defaultdict(int)
        self.invol_labels_with_clients = defaultdict(set)

        self.agg_type = args.agg_type
        self.evaluate_time = 0
        self.tot_train_samples = 0
        self.current_epoch = 0
        self.global_classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.buffersize = args.buffersize
        self.N_cloud = defaultdict(int)

    def set_clients(self, clientObj):
        # 加载数据集
        for i, train_slow, send_slow in zip(
            range(self.num_clients), self.train_slow_clients, self.send_slow_clients
        ):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            self.tot_train_samples += len(train_data)
            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=train_slow,
                send_slow=send_slow,
            )
            self.clients.append(client)

    # 只实现FedProto
    def set_edges(self, edgeObj):
        cids = np.arange(self.num_clients)
        for i in range(self.num_edges):
            # Randomly select clients and assign them
            selected_cids = np.random.choice(cids, self.clients_per_edge, replace=False)
            cids = list(set(cids) - set(selected_cids))
            self.edges.append(edgeObj(args=self.args, id=i, cids=selected_cids))
            [self.edges[i].client_register(self.clients[cid]) for cid in selected_cids]
            self.edges[i].all_trainsample_num = sum(
                self.edges[i].sample_registration.values()
            )
            self.edges[i].p_clients = [
                sample / float(self.edges[i].all_trainsample_num)
                for sample in list(self.edges[i].sample_registration.values())
            ]
            self.edges[i].refresh_edgeserver()
        # [self.edge_register(edge=edge) for edge in self.edges]
        self.p_edge = [
            sample / sum(self.sample_registration.values())
            for sample in list(self.sample_registration.values())
        ]

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_edges(self, return_all_edges=True):
        if return_all_edges is True:
            return self.edges

    def select_clients(self, client_losses=[], return_all_clients=False):
        if return_all_clients is True:
            return self.clients

        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(
            np.random.choice(self.clients, self.current_num_join_clients, replace=False)
        )
        return selected_clients

    def send_parameters(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            client.set_parameters()

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        #    https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221

    def proto_aggregation_clients(self):
        for id in self.uploaded_client_ids:
            clientprotos = load_item(
                self.client[id].role, "protos", self.clients[id].save_folder_name
            )
            save_item(
                clientprotos,
                self.client[id].role,
                "cloud_protos",
                self.clients[id].save_folder_name,
            )

            self.N_cloud[id] = self.clients[id].label_counts

        cloud_clientProtos = {
            client.id: load_item(client.role, "cloud_protos", client.save_folder_name)
            for client in self.clients
        }

        agg_protos_label = defaultdict(default_tensor)
        for j in range(self.num_classes):
            for id in cloud_clientProtos.keys():
                if (
                    cloud_clientProtos[id] is not None
                    and j in cloud_clientProtos[id].keys()
                ):
                    agg_protos_label[j] += (
                        self.N_cloud[id][j] * cloud_clientProtos[id][j]
                    )
            denominator = sum(
                self.N_cloud[id][j]
                for id, values in self.N_cloud.items()
                if j in values
            )
            agg_protos_label[j] = agg_protos_label[j] / denominator
        return agg_protos_label

    def proto_aggregation(self, edge_protos_list):
        agg_protos_label = defaultdict(default_tensor)
        if self.agg_type == 0:  # 按数据量平均
            for j in range(self.args.num_classes):
                for edge in self.edges:
                    id = edge.id
                    if (
                        id in self.uploaded_ids
                        and j in edge_protos_list[id]["protos"].keys()
                    ):
                        agg_protos_label[j] = agg_protos_label[j].to(self.device)
                        agg_protos_label[j] += (
                            self.edges[id].N_l[j] * edge_protos_list[id]["protos"][j]
                        )
                        self.edges[id].N_l_prev[j] = self.edges[id].N_l[j]
                        assert len(agg_protos_label[j]) == self.args.feature_dim
                    elif (
                        edge_protos_list[id]["prev_protos"] is not None
                        and j in edge_protos_list[id]["prev_protos"].keys()
                    ):
                        agg_protos_label[j] += (
                            self.edges[id].N_l_prev[j]
                            * edge_protos_list[id]["prev_protos"][j]
                        )
                        assert len(agg_protos_label[j]) == self.args.feature_dim

                if agg_protos_label[j] is not None:
                    agg_protos_label[j] = agg_protos_label[j] / sum(
                        edge.N_l_prev[j] for edge in self.edges
                    )

        print("agg_protos_label", agg_protos_label.keys())
        for id in self.uploaded_ids:
            # 更新edge的prev_protos
            if edge_protos_list[id] is not None:
                save_item(
                    edge_protos_list[id]["protos"],
                    self.edges[id].role,
                    "prev_protos",
                    self.save_folder_name,
                )
            # 更新client的prev_protos
            for client_id in self.edges[id].selected_cids:
                client_protos = load_item(
                    self.clients[client_id].role, "protos", self.save_folder_name
                )
                if client_protos is not None:
                    save_item(
                        client_protos,
                        self.clients[client_id].role,
                        "prev_protos",
                        self.save_folder_name,
                    )
        return agg_protos_label

    def aggregate_parameters(self):
        assert len(self.uploaded_ids) > 0

        client = self.clients[self.uploaded_ids[0]]
        global_model = load_item(client.role, "model", client.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()

        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, "model", client.save_folder_name)
            for server_param, client_param in zip(
                global_model.parameters(), client_model.parameters()
            ):
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, "global_model", self.save_folder_name)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc)
                hf.create_dataset("rs_test_auc", data=self.rs_test_auc)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss)

        if "temp" in self.save_folder_name:
            try:
                shutil.rmtree(self.save_folder_name)
                print("Deleted.")
            except:
                print("Already deleted.")

    def test_metrics(self):
        num_samples = []
        tot_regular_correct = []
        tot_proto_correct = []
        regular_accuracies = []
        proto_accuracies = []

        # 收集每个客户端的测试结果
        for c in self.clients:
            regular_acc, regular_num, proto_acc, proto_num = c.test_metrics()

            # 计算每个客户端的准确率
            regular_accuracy = regular_acc * 1.0 / regular_num if regular_num > 0 else 0
            proto_accuracy = proto_acc * 1.0 / proto_num if proto_num > 0 else 0

            # 记录准确率以便后续打印
            regular_accuracies.append((c.id, regular_accuracy))
            proto_accuracies.append((c.id, proto_accuracy))

            # 记录总数用于后续平均计算
            tot_regular_correct.append(regular_acc * 1.0)
            tot_proto_correct.append(proto_acc * 1.0)
            num_samples.append(regular_num)

        # 打印所有客户端的常规模型准确率
        # print("Regular Model Accuracies:")
        # for client_id, acc in regular_accuracies:
        # print(f"Client {client_id}: Regular Model Acc: {acc:.4f}")

        # 打印所有客户端的原型模型准确率
        # print("Prototype Model Accuracies:")
        # for client_id, acc in proto_accuracies:
        # print(f"Client {client_id}: Prototype Model Acc: {acc:.4f}")

        # 返回客户端ID、样本数量、常规模型和原型模型的准确数
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_regular_correct, tot_proto_correct

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
            print(f"Client {c.id}: Loss: {cl*1.0/ns}")

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()  # 调用 test_metrics 来收集所有统计信息

        # 解包 stats 中的统计信息
        ids, num_samples, tot_regular_correct, tot_proto_correct = stats

        # 计算常规模型的平均测试准确率
        regular_acc = (
            sum(tot_regular_correct) / sum(num_samples) if sum(num_samples) > 0 else 0
        )
        proto_acc = (
            sum(tot_proto_correct) / sum(num_samples) if sum(num_samples) > 0 else 0
        )

        # 打印平均准确率
        print("Averaged Test Accuracy (Regular Model): {:.4f}".format(regular_acc))
        print("Averaged Test Accuracy (Prototype Model): {:.4f}".format(proto_acc))

        # 计算标准差
        accs = [a / n for a, n in zip(tot_regular_correct, num_samples)]
        proto_accs = [p / n for p, n in zip(tot_proto_correct, num_samples)]

        # 打印标准差
        print("Std Test Accuracy (Regular Model): {:.4f}".format(np.std(accs)))
        print("Std Test Accuracy (Prototype Model): {:.4f}".format(np.std(proto_accs)))

        # 如果需要，记录测试准确率
        if acc is None:
            self.rs_test_acc.append(regular_acc)
        else:
            acc.append(regular_acc)

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def find_longest_time_client(self, first_client_epoch_times):
        # print("first_client_epoch_times:",first_client_epoch_times)
        longest_time_client = max(first_client_epoch_times, key=lambda x: x["time"])
        # print("longest_time_client",longest_time_client)
        return longest_time_client

    def get_total_labels_counts(self, local_aggs_list):
        self.total_counts = defaultdict(int)
        for local_agg in local_aggs_list:
            for label, count in local_agg.label_counts.items():
                self.total_counts[label] += count

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None

    def save_tsne_with_agg(
        self, args, base_path, drawtype="clientavgproto", current_epoch=0
    ):
        """
        生成并保存包含本地和聚合原型的 t-SNE 图。
        """
        if args.algorithm == "FedSAE":
            save_folder = f"{base_path}/{args.dataset}/{args.algorithm}/localepoch_{args.local_epochs}/agg_{args.agg_type}/buffer_{args.buffersize}/lamda_{args.lamda}/addTGP_{args.addTGP}_gl_use_clients_{args.gl_use_clients}_gamma_{args.gamma}_beta_{args.SAEbeta}_usegl_{args.test_useglclassifier}_lr_{args.local_learning_rate}/{drawtype}"
        else:
            save_folder = f"{base_path}/{args.dataset}/{args.algorithm}/localepoch_{args.local_epochs}_agg_{args.agg_type}_lamda_{args.lamda}_lr_{args.local_learning_rate}/{drawtype}"

        all_features = []
        all_labels = []
        label_types = []  # 区分来源：agg 或 local

        global_protos = load_item("Server", "global_protos", self.save_folder_name)
        # global_protos = load_item(self.role, "global_protos", self.save_folder_name)
        # 收集聚合原型数据
        for label, proto in global_protos.items():
            all_features.append(proto.cpu().detach().numpy())
            all_labels.append(label)
            label_types.append("Global")  # 标记为全局原型

        # 收集本地原型数据
        for client in self.clients:
            if drawtype == "clientavgproto":
                client_protos = load_item(
                    client.role, "prev_protos", client.save_folder_name
                )
                if client_protos is not None:
                    for key in client_protos.keys():
                        all_features.append(client_protos[key].cpu().detach().numpy())
                        all_labels.append(key)
                        label_types.append("Local")  # 标记为本地原型
            elif drawtype == "clientallfeatures":
                client_features = load_item(
                    client.role, "featureSet", client.save_folder_name
                )
                if client_features is not None:
                    for key in client_features.keys():
                        for tensor in client_features[key]:
                            # 转换每个 tensor 为 NumPy 数组后添加到 all_features
                            all_features.append(tensor.detach().cpu().numpy())
                        all_labels.extend([label] * len(client_features[key]))
                        label_types.extend(
                            ["Local"] * len(client_features[key])
                        )  # 标记为本地原型

        # 转为 NumPy 数组
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        print("Number of samples:", len(all_features))

        # t-SNE 降维，调整 perplexity
        # perplexity = min(30, len(all_features) - 1)
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(all_features)

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
        # plt.legend()
        plt.title(
            f"t-SNE Visualization with Global and Local Prototypes (Epoch {current_epoch})"
        )
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.savefig(save_path)
        plt.close()
