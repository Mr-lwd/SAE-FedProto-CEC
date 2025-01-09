import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.func_utils import generate_and_plot_umap
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit
from collections import defaultdict
import random
import math
import json
from utils.io_utils import load_item, save_item


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
        self.num_workers = self.args.num_workers

        # 创建client model
        if args.goal == "test" and (args.save_folder_name == "temp" or "temp" not in args.save_folder_name):
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
        self.trans_simu_time = random.randint(1, 10)
        self.receive_buffer = None
        self.optimizer = self.args.optimizer
        self.local_model_loss = 0
        self.local_all_loss = 0
        if self.args.jetson == 1:
            self.dvfs_data = self.create_objects_from_json()
            self.maxCPUfreq = max([item["frequency"] for item in self.dvfs_data])

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(
            train_data,
            batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        if self.args.goal == "gltest":
            # Directly load the full test dataset
            test_data_dir = "../dataset"
            from torchvision import datasets, transforms

            if "FashionMNIST" in self.args.dataset:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                )
                test_data = datasets.FashionMNIST(
                    root=f"{test_data_dir}/FashionMNIST",
                    train=False,
                    download=False,
                    transform=transform,
                )
                print(f"{self.args.dataset},len(test_data): {len(test_data)}")
            elif "Cifar10" in self.args.dataset:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
                test_data = datasets.CIFAR10(
                    root=test_data_dir, train=False, download=False, transform=transform
                )
                print(f"{self.args.dataset},len(test_data): {len(test_data)}")
            elif "MNIST" in self.args.dataset:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                )
                test_data = datasets.MNIST(
                    root=f"{test_data_dir}/MNIST/rawdata",
                    train=False,
                    download=False,
                    transform=transform,
                )
                print(f"{self.args.dataset},len(test_data): {len(test_data)}")
            else:
                raise ValueError(
                    f"Dataset {self.args.dataset} not supported for global testing"
                )
        else:
            # Original code for client-specific testing
            test_data = read_client_data(self.dataset, self.id, is_train=False)

        return DataLoader(
            test_data, batch_size, drop_last=False, shuffle=False, num_workers=0
        )

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics_proto(self):
        testloader = self.load_test_data()
        model = load_item(self.role, "model", self.save_folder_name)
        global_protos = load_item("Server", "global_protos", self.save_folder_name)
        if self.args.test_useglclassifier == 1:
            client_classifier = model.head  # 假设客户端分类器存储在 head 属性
            glclassifier = load_item("Server", "glclassifier", self.save_folder_name)
            if glclassifier is not None:
                client_classifier.load_state_dict(glclassifier.state_dict())
        if self.args.DVFS == 1:
            model = model.to("cuda")
            for label, tensor in global_protos.items():
                if isinstance(tensor, torch.Tensor):  # 确认值是 PyTorch 张量
                    global_protos[label] = tensor.to("cuda")
                else:
                    raise TypeError(
                        f"Value for label '{label}' is not a torch.Tensor. Type: {type(tensor)}"
                    )
        else:
            model = model.to(self.device)

        # global_protos = load_item("Server", "global_protos", self.save_folder_name)
        model.eval()

        # Regular inference accuracy (baseline accuracy using the model alone)
        regular_acc = 0
        regular_num = 0
        proto_acc = 0
        proto_num = 0
        X = []
        Y = []

        # Regular model inference
        if global_protos is not None:
            with torch.no_grad():
                correct_class_count_regular = {
                    cls: 0 for cls in range(self.num_classes)
                }
                correct_class_count_proto = {cls: 0 for cls in range(self.num_classes)}
                for images, labels in testloader:
                    if self.args.DVFS == 1:
                        images, labels = images.to("cuda"), labels.to("cuda")
                    else:
                        images, labels = images.to(self.device), labels.to(self.device)

                    # Regular model inference
                    rep = model.base(images)
                    outputs = model.head(rep)
                    outputs = outputs.squeeze(1)  # Remove the extra dimension

                    # Calculate correct predictions for regular model
                    _, pred_labels = torch.max(outputs, dim=1)
                    pred_labels = pred_labels.view(-1)
                    regular_acc += torch.sum(pred_labels == labels).item()
                    regular_num += len(labels)

                    if self.args.goal == "gltest_umap":
                        rep=rep.squeeze(1)
                        X.extend(rep.cpu().numpy())
                        Y.extend(labels.cpu().numpy())

                    for label in labels[pred_labels == labels].tolist():
                        correct_class_count_regular[label] += 1
                    # Prototype-based inference accuracy (using global_protos)
                    if global_protos is not None:
                        rep = model.base(
                            images
                        )  # Extract the representation for prototypes
                        if self.args.DVFS == 1:
                            output = float("inf") * torch.ones(
                                labels.shape[0], self.num_classes
                            ).to("cuda")
                        else:
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
            if self.args.goal == "gltest":
                print("-" * 30)
                print(f"client id {self.id}")
                print("Regular Model Correct Classifications:")
                print(correct_class_count_regular)
                print("Prototype-Based Model Correct Classifications:")
                print(correct_class_count_proto)
            if self.args.goal == "gltest_umap":
                features = {}
                features["X"] = X
                features["Y"] = Y
                print(f"X shape: {np.array(X).shape}, Y shape: {np.array(Y).shape}")
                save_item(features, self.role, "umap_features", self.save_folder_name)
            return regular_acc, regular_num, proto_acc, proto_num
        else:
            return 0, 1e-5, 0, 1e-5

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
                output = output.squeeze(1)

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
        for _, (x, y) in enumerate(trainloader):
            for label in y:
                self.label_counts[label.item()] += 1
                total_labels += 1
        entropy = 0.0
        for label, count in self.label_counts.items():
            p_j = count / total_labels
            if p_j > 0:
                entropy -= p_j * math.log(p_j)
        self.entropy = entropy
        print("id", self.id)
        print("label_counts:", self.label_counts)
        # print("entropy", entropy)

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(
            client_id=self.id,
            cshared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict()),
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

    def create_objects_from_json(
        self, file_path="./DVFS/mutibackpack_algo/extracted_data.json"
    ):
        objects = None
        with open(file_path, "r") as file:
            objects = json.load(file)
        return objects


def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    file_path = os.path.join(item_path, role + "_" + item_name + ".pt")
    torch.save(item, file_path)

    # 查看保存后的文件大小（单位：字节）
    # if item_name == "CCVR":
    #     file_size = os.path.getsize(file_path)
    #     print(f"Saved file size: {file_size} bytes")

    #     # 如果你希望以 KB 或 MB 显示，可以做以下转换：
    #     file_size_kb = file_size / 1024
    #     file_size_mb = file_size_kb / 1024
    #     print(f"File size: {file_size_kb:.2f} KB")
    #     print(f"File size: {file_size_mb:.2f} MB")


def load_item(role, item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, role + "_" + item_name + ".pt"))
    except FileNotFoundError:
        print(role, item_name, "Not Found")
        return None
