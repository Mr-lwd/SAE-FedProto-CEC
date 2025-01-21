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
from .multibp_algo import get_dvfs_set
import json
import ctypes 

class clientSAE_DVFS(Client):
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
        self.leave_frequency_set=[]
        self.leave_local_epochs = self.local_epochs - 1
        self.storeoptimizer = None
        
        if self.args.jetson == 1:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, 'change_config_noprint.so')
            # 加载共享库
            print("lib_path", lib_path)
            self.cLib = ctypes.CDLL(lib_path)

    def train(self, firstlocaltrain=False, longest_time=0):
        self.receive_from_edgeserver()
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        global_protos = load_item("Server", "global_protos", self.save_folder_name)
        # if self.args.addTGP == 1:
        #     global_protos = load_item(
        #         "Server", "tgp_global_protos", self.save_folder_name
        #     )
        glclassifier = load_item("Server", "glclassifier", self.save_folder_name)
        if glclassifier is not None:  # 固定参数
            glclassifier.to(self.device)
            for param in glclassifier.parameters():
                param.requires_grad = False
        # self.client_protos = load_item(self.role, "protos", self.save_folder_name)
        # if firstlocaltrain is True:
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,momentum=self.args.momentum)
        model.to(self.device)
        model.train()
        
        if firstlocaltrain is True:
            self.leave_frequency_set=[]
            time.sleep(1)
            self.cLib.changeCpuFreq(self.maxCPUfreq)
            time.sleep(1)  
            

        if firstlocaltrain is False:
            self.leave_train_time = longest_time * self.leave_local_epochs
            # print("self.first_localepoch_time", self.first_localepoch_time)
            # print("longest_time", longest_time)
            if (self.first_localepoch_time < longest_time - 0.01):
                self.leave_frequency_set = get_dvfs_set(self.dvfs_data, self.first_localepoch_time, self.leave_train_time, self.leave_local_epochs)
                print(f"client {self.id} leave_frequency_set: {self.leave_frequency_set}")
                if self.leave_frequency_set == []:
                    print("hard to get frequency set, max frequency")
            else:
                self.leave_frequency_set = []
            
            if self.leave_frequency_set == []:
                time.sleep(1)
                self.cLib.changeCpuFreq(self.maxCPUfreq)
                time.sleep(1)  

        # max_local_epochs = self.local_epochs
        # if self.train_slow:
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        leave_freq_counter = 0
        sleepTime = 0
        
        if firstlocaltrain is False:
            if(self.leave_frequency_set!=[]):
                time.sleep(1)
                self.cLib.changeCpuFreq(self.leave_frequency_set[leave_freq_counter])
                time.sleep(1) 
                # print("frequency scale:",self.leave_frequency_set[leave_freq_counter])
                leave_freq_counter += 1  
        if self.args.jetson == 1:
            pl = PowerLogger(interval=3.0, nodes=getNodesByName(['module/cpu']))
            pl.start()
        
        local_train_start_time = time.perf_counter()  # 记录训练开始的时间
        for step in range(self.leave_local_epochs if firstlocaltrain is False else 1):   
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
                rep = rep.squeeze(1)
                # output = model.head(rep)
                if self.args.gamma < 1 or glclassifier is None:
                    output = model.head(rep)
                    if self.args.jetson == 1:
                        output = output.double()
                # loss = self.loss(output, y)
                if glclassifier is not None:
                    if self.args.gamma < 1:
                        loss = self.loss(output, y) * (1 - self.args.gamma)
                        global_outputs = glclassifier(rep)
                        if self.args.jetson == 1:
                            global_outputs = global_outputs.double()
                        global_loss = self.loss(global_outputs, y) * self.args.gamma
                        loss += global_loss
                    else: # gamma == 1
                        global_outputs = glclassifier(rep)
                        if self.args.jetson == 1:
                            global_outputs = global_outputs.double()
                        loss = self.loss(global_outputs, y)
                else:
                    loss = self.loss(output, y)
                self.local_model_loss += loss.item()

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                self.local_all_loss += loss.item()
                
                if firstlocaltrain is False and step == self.leave_local_epochs - 1:
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if firstlocaltrain is False:
                if(self.leave_frequency_set!=[]) and leave_freq_counter < len(self.leave_frequency_set):
                    time.sleep(1)
                    self.cLib.changeCpuFreq(self.leave_frequency_set[leave_freq_counter])
                    time.sleep(1) 
                    leave_freq_counter += 1  
                    sleepTime+=2
                
        if self.device == "cuda":
            torch.cuda.synchronize()
        local_train_time = time.perf_counter() - local_train_start_time - sleepTime
        if self.args.jetson == 1:
            pl.stop()
            averagePower = pl.getAveragePower(nodeName='module/cpu')  # 获取平均功耗
            self.energy += local_train_time * averagePower/1e3 #s * w = J
            print(f"client {self.id}, train time: {local_train_time}, power: {averagePower}")
        self.local_model_loss = self.local_model_loss / len(trainloader)
        self.local_all_loss =self.local_all_loss / len(trainloader)
        # print("local_model_loss", local_model_loss.item())
        # save_item(copy.deepcopy(protos), self.role, "featureSet", self.save_folder_name)
        self.train_time = local_train_time
        
        save_item(model, self.role, "model", self.save_folder_name)
        
        if firstlocaltrain is True:
            self.first_localepoch_time = self.train_time
            return self.id, self.train_time, self.trans_time
        
        self.cal_mean_and_covariance(protos)
        if self.args.drawGMM == 1:
            save_item(protos, self.role, "features", self.save_folder_name)
        agg_protos = self.agg_func(protos)
        save_item(agg_protos, self.role, "protos", self.save_folder_name)
        
        if self.trans_delay_simulate is True:
            self.trans_time += self.trans_simu_time
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
        ##接受来自服务器几乎没有延迟，只有平均原型和全局分类器
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

    def cal_mean_and_covariance(self, protos):
        """
        计算加权均值和协方差矩阵。
        保存均值和协方差矩阵。

        Args:
            protos (dict): {label: proto_list}, 其中
                           proto_list 是 512 维向量的列表。
        Returns:
            mean_dict (dict): {label: mean_vector}, 均值向量。
            cov_dict (dict): {label: covariance_matrix}, 协方差矩阵。
            counts (dict): {label: count}, 每个类别的样本数量。
        """
        mean_dict = defaultdict(np.ndarray)  # 保存每个类别的均值
        cov_dict = defaultdict(np.ndarray)   # 保存每个类别的协方差矩阵
        counts = defaultdict(int)  # 保存每个类别的样本数量

        for label, proto_list in protos.items():
            # 将原型列表转换为矩阵 (n_samples, 512)
            proto_array = np.array([proto.cpu().numpy() for proto in proto_list])  # (n_samples, 512)
            # proto_array = np.array([proto.numpy() for proto in proto_list])  # (n_samples, 512)

            # 计算加权均值
            mean_vector = np.mean(proto_array, axis=0)  # (512,)
            mean_dict[label] = mean_vector

            # 计算协方差矩阵
            centered_array = proto_array - mean_vector  # 每个向量减去均值
            covariance_matrix = np.cov(centered_array, rowvar=False)  # (512, 512)
            cov_dict[label] = covariance_matrix
            counts[label] = proto_array.shape[0]

        combined_meancov = prepare_item(mean_dict, cov_dict, counts)

        # 保存
        save_item(combined_meancov, self.role, "mean_cov", self.save_folder_name)
        # return mean_dict, cov_dict
            

    # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].detach()
                for i in proto_list:
                    proto += i.detach()
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0].detach()
            # 平滑

        return protos
    
    # def test_metrics_proto(self):
    #     testloader = self.load_test_data()
    #     model = load_item(self.role, "model", self.save_folder_name)
    #     if self.args.test_useglclassifier == 1:
    #         client_classifier = model.head  # 假设客户端分类器存储在 head 属性
    #         glclassifier = load_item("Server", "glclassifier", self.save_folder_name)
    #         if glclassifier is not None:
    #             client_classifier.load_state_dict(glclassifier.state_dict())
                
    #     global_protos = load_item("Server", "global_protos", self.save_folder_name)
    #     model = model.to(self.device)
    #     model.eval()

    #     # Regular inference accuracy (baseline accuracy using the model alone)
    #     regular_acc = 0
    #     regular_num = 0
    #     proto_acc = 0
    #     proto_num = 0

    #     # Regular model inference
    #     if global_protos is not None:
    #         with torch.no_grad():
    #             correct_class_count_regular = {
    #                 cls: 0 for cls in range(self.num_classes)
    #             }
    #             correct_class_count_proto = {cls: 0 for cls in range(self.num_classes)}
    #             for images, labels in testloader:
    #                 images, labels = images.to(self.device), labels.to(self.device)

    #                 # Regular model inference
    #                 outputs = model(images)
    #                 outputs = outputs.squeeze(1)  # Remove the extra dimension

    #                 # Calculate correct predictions for regular model
    #                 _, pred_labels = torch.max(outputs, dim=1)
    #                 pred_labels = pred_labels.view(-1)
    #                 regular_acc += torch.sum(pred_labels == labels).item()
    #                 regular_num += len(labels)

    #                 for label in labels[pred_labels == labels].tolist():
    #                     correct_class_count_regular[label] += 1
    #                 # Prototype-based inference accuracy (using global_protos)
    #                 if global_protos is not None:
    #                     rep = model.base(
    #                         images
    #                     )  # Extract the representation for prototypes
    #                     output = float("inf") * torch.ones(
    #                         labels.shape[0], self.num_classes
    #                     ).to(self.device)

    #                     for i, r in enumerate(rep):
    #                         for j, pro in global_protos.items():
    #                             if type(pro) != type([]):
    #                                 output[i, j] = self.loss_mse(r, pro)
    #                     proto_predictions = torch.argmin(output, dim=1)
    #                     proto_acc += torch.sum(proto_predictions == labels).item()
    #                     proto_num += len(labels)
    #                     for label in labels[proto_predictions == labels].tolist():
    #                         correct_class_count_proto[label] += 1

    #                     # 打印统计结果
    #         if self.args.goal == "gltest":
    #             print("-" * 30)
    #             print(f"client id {self.id}")
    #             print("Regular Model Correct Classifications:")
    #             print(correct_class_count_regular)
    #             print("Prototype-Based Model Correct Classifications:")
    #             print(correct_class_count_proto)
    #         return regular_acc, regular_num, proto_acc, proto_num
    #     else:
    #         return 0, 1e-5, 0, 1e-5


def prepare_item(mean_dict, cov_dict, counts):
    """
    将均值和协方差转为 Tensor 并合并为字典对象。
    
    Args:
        mean_dict (dict): {label: mean_vector}, 每个类别的均值。
        cov_dict (dict): {label: covariance_matrix}, 每个类别的协方差矩阵。
        counts (dict): {label: count}, 每个类别的样本数量。
    Returns:
        combined_dict (dict): {label: {"mean": Tensor, "cov": Tensor, "counts": int}}, 合并后的字典。
    """
    combined_dict = {}
    for label in mean_dict.keys():
        combined_dict[label] = {
            "mean": torch.tensor(mean_dict[label], dtype=torch.float32),
            "cov": torch.tensor(cov_dict[label], dtype=torch.float32),
            "counts": counts[label]
        }
    return combined_dict