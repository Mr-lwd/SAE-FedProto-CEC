import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit
from collections import defaultdict
import random
import math


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.role = "Client_" + str(self.id)
        self.save_folder_name = args.save_folder_name_full

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.args = args
        self.involve_count = 0

        # 创建client model
        if args.save_folder_name == "temp" or "temp" not in args.save_folder_name:
            model = BaseHeadSplit(args, self.id).to(self.device)
            save_item(model, self.role, "model", self.save_folder_name)

        self.train_slow = kwargs["train_slow"]
        self.send_slow = kwargs["send_slow"]
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        self.loss = nn.CrossEntropyLoss()

        self.label_counts = defaultdict(int)
        self.entropy = 0
        self.initLabels()
        self.trans_delay_simulate = args.trans_delay_simulate
        self.sleep_time = random.randint(1, 10)
        self.receive_buffer = None
        

    def load_train_data(self, batch_size=None, num_workers=4):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(
            train_data,
            batch_size,
            drop_last=False,
            shuffle=False,
        )

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, "model", self.save_folder_name)
        # model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
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
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def initLabels(self):
        trainloader = self.load_train_data()
        total_labels = 0
        # label_counts = defaultdict(int)
        for _, (x, y) in enumerate(trainloader):
            for label in y:
                self.label_counts[label.item()] += 1
                total_labels += 1
            # 计算熵值 H(C_i)
        entropy = 0.0
        for label, count in self.label_counts.items():
            # 计算每个标签的概率 p_j
            p_j = count / total_labels
            if p_j > 0:
                entropy -= p_j * math.log(p_j)
        self.entropy = entropy
        print("id", self.id)
        print("self.label_counts", self.label_counts)
        print("entropy", entropy)
        
    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
                                        )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None

def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))


def load_item(role, item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, role + "_" + item_name + ".pt"))
    except FileNotFoundError:
        print(role, item_name, "Not Found")
        return None
