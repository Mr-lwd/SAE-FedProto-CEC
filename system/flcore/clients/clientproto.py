import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.io_utils import load_item, save_item
from utils.func_utils import *
from collections import defaultdict
from .measure_power import *
import json


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.cshared_protos_local = {}
        self.cshared_protos_global = None
        self.train_time = 0
        self.trans_time = 0


    def train(self):
        self.receive_from_edgeserver()
        # print("Client.id begin training", self.id)
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        global_protos = load_item("Server", "global_protos", self.save_folder_name)
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        model.to(self.device)
        model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        local_train_start_time = time.perf_counter()  # 记录训练开始的时间
        if self.args.jetson == 1:
            pl = PowerLogger(interval=3.0, nodes=getNodesByName(['module/cpu']))
        for step in range(max_local_epochs):
            if self.args.jetson == 1 and step == 1:
                pl.start()
            self.local_model_loss = 0
            self.local_all_loss = 0
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
                # print("rep", rep)
                rep = rep.squeeze(1)
                output = model.head(rep)
                if self.args.jetson == 1:
                    output = output.double()
                loss = self.loss(output, y)
                # print("loss", loss.item())
                self.local_model_loss += loss.item()
                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                self.local_all_loss += loss.item()
                if step == max_local_epochs - 1:
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if self.device == "cuda":
            torch.cuda.synchronize()
        local_train_time = time.perf_counter() - local_train_start_time
        if self.args.jetson == 1:
            pl.stop()
            averagePower = pl.getAveragePower(nodeName='module/cpu')  # 获取平均功耗
            self.energy += local_train_time * averagePower/1e3 #s * w = J
        self.local_model_loss = self.local_model_loss / len(trainloader)
        self.local_all_loss = self.local_all_loss / len(trainloader)
        
        # print("local_model_loss", self.local_model_loss)
        # print("local_gl_loss", local_gl_loss)\
        # 计算每个 512 维张量的大小（单位：字节）
        # 计算字典中所有张量的总大小（单位：字节）

        # all_bytes = 0
        # all_bytes += get_theory_bytes(protos)
        
        # all_bytes += get_theory_bytes(agg_protos)
        # print(f"Tensor size: {all_bytes} bytes")
        
        agg_protos = self.agg_func(protos)
        save_item(agg_protos, self.role, "protos", self.save_folder_name)
        save_item(model, self.role, "model", self.save_folder_name)

        self.train_time = local_train_time
        # if self.trans_delay_simulate is True:
        #     self.trans_time += self.trans_simu_time
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += local_train_time
        # c^l_i, X^l_i直接从本地读取，self.role = "Client_"+str(self.id)
        return self.id, self.train_time, self.trans_time


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
        # if self.trans_delay_simulate is True:
        #     self.trans_time += self.trans_simu_time
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
