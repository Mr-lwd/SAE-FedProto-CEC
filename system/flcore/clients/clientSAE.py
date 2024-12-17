import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from utils.func_utils import *
from collections import defaultdict


class clientSAE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.cshared_protos_local = {}
        self.cshared_protos_global = None
        self.featureAndlabels = None
        self.train_time = 0
        self.trans_time = 0

    def train(self):
        self.receive_from_edgeserver()
        # print("Client.id begin training", self.id)
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        global_protos = load_item("Server", "global_protos", self.save_folder_name)
        if self.args.addTGP:
            tgp_global_protos = load_item(
                "Server", "tgp_global_protos", self.save_folder_name
            )
        glclassifier = load_item("Server", "glclassifier", self.save_folder_name)
        if glclassifier is not None:  # 固定参数
            for param in glclassifier.parameters():
                param.requires_grad = False

            if self.args.mixclassifier == 1:
                client_classifier = model.head
                client_state_dict = client_classifier.state_dict()
                global_state_dict = glclassifier.state_dict()
                averaged_state_dict = {}
                for key in client_state_dict.keys():
                    if key in global_state_dict:
                        # averaged_state_dict[key] = (
                        #     client_state_dict[key] + global_state_dict[key]
                        # ) / 2
                        averaged_state_dict[key] = (
                            0.7 * client_state_dict[key] + 0.3 * global_state_dict[key]
                        )
                    else:
                        averaged_state_dict[key] = client_state_dict[key]

                client_classifier.load_state_dict(averaged_state_dict)

        # print("local global protos", global_protos)
        self.client_protos = load_item(self.role, "protos", self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        local_train_start_time = time.perf_counter()  # 记录训练开始的时间
        for step in range(max_local_epochs):
            local_model_loss = 0
            protos = defaultdict(list)
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                rep = rep.squeeze(1)
                output = model.head(rep)
                # loss = self.loss(output, y)
                if glclassifier is not None and self.args.mixclassifier != 1:
                    loss = self.loss(output, y) * (1 - self.args.gamma)
                    global_outputs = glclassifier(rep)
                    global_loss = self.loss(global_outputs, y) * self.args.gamma
                    loss += global_loss
                else:
                    loss = self.loss(output, y)

                local_model_loss += loss
                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    if self.args.use_beta:
                        loss += (
                            self.loss_mse(proto_new, rep)
                            * self.lamda
                            * (1 - self.args.SAEbeta)
                        )
                    else:
                        loss += self.loss_mse(proto_new, rep) * self.lamda
                if self.args.use_beta and tgp_global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(tgp_global_protos[y_c]) != type([]):
                            proto_new[i, :] = tgp_global_protos[y_c].data
                    loss += (
                        self.loss_mse(proto_new, rep) * self.lamda * self.args.SAEbeta
                    )
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if self.device == "cuda":
            torch.cuda.synchronize()
        local_train_time = time.perf_counter() - local_train_start_time
        local_model_loss = local_model_loss / len(trainloader)
        # print("local_model_loss", local_model_loss.item())
        # save_item(copy.deepcopy(protos), self.role, "featureSet", self.save_folder_name)
        agg_protos = self.agg_func(protos)
        save_item(agg_protos, self.role, "protos", self.save_folder_name)
        save_item(model, self.role, "model", self.save_folder_name)

        self.train_time = local_train_time
        if self.trans_delay_simulate is True:
            self.trans_time += self.trans_simu_time
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += local_train_time
        # c^l_i, X^l_i直接从本地读取，self.role = "Client_"+str(self.id)
        return self.id, self.train_time, self.trans_time

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, "model", self.save_folder_name)
        if self.args.test_useglclassifier == 1:
            client_classifier = model.head  # 假设客户端分类器存储在 head 属性
            glclassifier = load_item("Server", "glclassifier", self.save_folder_name)
            if glclassifier is not None:
                client_classifier.load_state_dict(glclassifier.state_dict())
            # print("g_classifier test_metrics")
        model = model.to(self.device)
        if self.args.addTGP:
            global_protos = load_item(
                "Server", "tgp_global_protos", self.save_folder_name
            )
        else:
            global_protos = load_item("Server", "global_protos", self.save_folder_name)
        model.eval()

        # Regular inference accuracy (baseline accuracy using the model alone)
        regular_acc = 0
        regular_num = 0
        proto_acc = 0
        proto_num = 0

        # Regular model inference
        if global_protos is not None:
            with torch.no_grad():
                correct_class_count_regular = {
                    cls: 0 for cls in range(self.num_classes)
                }
                correct_class_count_proto = {cls: 0 for cls in range(self.num_classes)}
                for images, labels in testloader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Regular model inference
                    outputs = model(images)
                    outputs = outputs.squeeze(1)  # Remove the extra dimension

                    # Calculate correct predictions for regular model
                    _, pred_labels = torch.max(outputs, dim=1)
                    pred_labels = pred_labels.view(-1)
                    regular_acc += torch.sum(pred_labels == labels).item()
                    regular_num += len(labels)

                    for label in labels[pred_labels == labels].tolist():
                        correct_class_count_regular[label] += 1
                    # Prototype-based inference accuracy (using global_protos)
                    if global_protos is not None:
                        rep = model.base(
                            images
                        )  # Extract the representation for prototypes
                        output = float("inf") * torch.ones(
                            labels.shape[0], self.num_classes
                        ).to(self.device)

                        for i, r in enumerate(rep):
                            for j, pro in global_protos.items():
                                if type(pro) != type([]):
                                    output[i, j] = self.loss_mse(r, pro)
                        proto_predictions = torch.argmin(output, dim=1)
                        proto_acc += torch.sum(proto_predictions == labels).item()
                        proto_num += len(labels)
                        for label in labels[proto_predictions == labels].tolist():
                            correct_class_count_proto[label] += 1

                        # 打印统计结果
                # print("-" * 30)
                # print(f"client id {self.id}")
                # print("Regular Model Correct Classifications:")
                # print(correct_class_count_regular)
                # print("Prototype-Based Model Correct Classifications:")
                # print(correct_class_count_proto)
            return regular_acc, regular_num, proto_acc, proto_num
        else:
            return 0, 1e-5, 0, 1e-5

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        global_protos = load_item("Server", "global_protos", self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                    # 原型距离损失：计算 rep 与其他标签原型的距离
        return losses, train_num

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(
            client_id=self.id,
            cshared_protos_local=copy.deepcopy(self.cshared_protos_local),
        )
        return None

    def receive_from_edgeserver(self, eshared_protos_global=None):
        # client
        self.train_time = 0
        self.trans_time = 0
        if self.trans_delay_simulate is True:
            self.trans_time += self.trans_simu_time
        # self.receive_buffer = eshared_protos_global
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.cshared_protos_global = self.receive_buffer
        return None

    # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data.detach()
                for i in proto_list:
                    proto += i.data.detach()
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0].detach()
            # 平滑

        return protos
