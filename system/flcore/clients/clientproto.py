import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
import random
import os
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import StepLR


class FocalLoss_MaxScaling(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="none"):
        """
        初始化Focal Loss。
        alpha: 平衡因子，如果没有类别不平衡，可以设置为1。
        gamma: 聚焦参数，默认为2。
        reduction: 损失的归约方式，可以是'none'、'mean'或'sum'。
        """
        super(FocalLoss_MaxScaling, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.client_protos = None

    def forward(self, inputs, targets):
        """
        计算Focal Loss。
        inputs: 模型的预测值，假设为(logits)的形式，大小为(N, C)。
        targets: 目标类别标签，大小为(N)。
        """
        # 限制 logits 范围，防止梯度爆炸
        inputs = torch.clamp(inputs, min=-10, max=10)

        # 计算 softmax 概率
        prob = F.softmax(inputs, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)  # 避免 pt 为 0 或 1

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        # 计算每个样本的预测概率
        # pt = torch.exp(-ce_loss)  # pt是预测正确类别的概率
        # 计算Focal Loss

        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLossOptimized(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLossOptimized, self).__init__()
        self.margin = margin

    def forward(self, reps, labels):
        """
        reps: 样本的嵌入特征，形状为 [batch_size, feature_dim]
        labels: 样本对应的标签，形状为 [batch_size]
        """
        # 计算样本对的余弦相似度
        cos_sim = F.cosine_similarity(reps.unsqueeze(0), reps.unsqueeze(1), dim=2)
        # print(reps[0])
        # exit(0)
        # 是否同类的布尔矩阵
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 同类样本之间的聚拢损失
        same_class_loss = label_matrix * torch.pow(1 - cos_sim, 2)
        print("same_class_loss", same_class_loss)
        # 不同类样本之间的分离损失
        diff_class_loss = (~label_matrix) * torch.pow(
            torch.clamp(self.margin - cos_sim, min=0), 2
        )
        print("cos_sim")
        print("diff_class_loss", diff_class_loss)
        # 计算总损失并返回
        loss = same_class_loss + diff_class_loss
        return loss.mean()


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

    def train(self, glprotos_invol_dataset, global_classifier=None, global_epoch=0):
        print("client.id trained", self.id)
        # 0.5 effect
        # contrastive_loss_fn = ContrastiveLossOptimized(margin=0.5)
        if self.args.use_focalLoss == 1:
            loss_fn = FocalLoss_MaxScaling(alpha=1, gamma=2, reduction="mean")
        start_time = time.perf_counter()
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        global_protos = load_item("Server", "global_protos", self.save_folder_name)
        if self.args.agg_type == 4:
            avg_global_protos = load_item(
                "Server", "avg_global_protos", self.save_folder_name
            )
        self.client_protos = load_item(self.role, "protos", self.save_folder_name)
        # reverse_global_protos = load_item('Server', 'reverse_global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # 设置每隔多少个epoch衰减学习率以及衰减的倍率
        # if self.args.use_scheduler:
        #     # print("use scheduler lr decay")
        #     scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        # model.to(self.device)
        model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # protos = defaultdict(list)

        for step in range(max_local_epochs):
            local_model_loss = 0
            local_gl_loss = 0
            protos = defaultdict(list)
            # protos_true = defaultdict(list)
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

                if global_epoch > 0 and self.args.glclassifier == 1:
                    loss = self.loss(output, y) * (1 - self.args.gamma)
                else:
                    loss = self.loss(output, y)

                if global_epoch > 0 and self.args.glclassifier == 1:
                    # 2. 如果全局分类器存在，计算指导损失
                    # 转为概率分布
                    for param in global_classifier.parameters():
                        param.requires_grad = False

                    global_outputs = global_classifier(rep)
                    if self.args.use_focalLoss == 1:
                        global_loss = loss_fn(global_outputs, y)
                    else:
                        global_loss = self.loss(global_outputs, y)

                    loss += global_loss * self.args.gamma
                    local_gl_loss += global_loss * self.args.gamma

                local_model_loss += loss
                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # if self.args.use_scheduler:
            #     scheduler.step()  # 学习率衰减
            # if self.args.each_epoch_update_proto == 1 and global_protos is not None:
            #     temp_protos = copy.deepcopy(protos)
            #     new_local_protos = self.agg_func(temp_protos)
            #     global_protos = self.update_global_protos(
            #         new_local_protos, global_protos, glprotos_invol_dataset
            #     )
            #     # 更新本地模型
            # self.client_protos = new_local_protos
        if self.device == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_time
        local_model_loss = local_model_loss / len(trainloader)
        local_gl_loss = local_gl_loss / len(trainloader)
        print("local_model_loss", local_model_loss)
        print("local_gl_loss", local_gl_loss)
        client_protos_data = copy.deepcopy(protos)
        # client_protos = copy.deepcopy(protos)
        # client_protos_data = self.agg_func(client_protos)
        save_item(self.agg_func(protos), self.role, "protos", self.save_folder_name)
        save_item(model, self.role, "model", self.save_folder_name)

        train_time = epoch_time

        if self.trans_delay_simulate is True:
            epoch_time += self.sleep_time * 2
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += epoch_time
        return train_time, epoch_time, client_protos_data
        # print('total_cost:', self.train_time_cost['total_cost'])

    def test_metrics(self, g_classifier=None):
        testloader = self.load_test_data()
        model = load_item(self.role, "model", self.save_folder_name)
        if (
            g_classifier is not None
            and self.args.glclassifier == 1
            and self.args.test_useglclassifier == 1
        ):
            client_classifier = model.head  # 假设客户端分类器存储在 head 属性
            client_classifier.load_state_dict(g_classifier.state_dict())
            print("g_classifier test_metrics")
            # model.head.weight.data = g_classifier.weight.data
            # model.head.bias.data = g_classifier.bias.data
        model = model.to(self.device)
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
                print("-" * 20)
                print(f"client id {self.id}")
                print("Regular Model Correct Classifications (per class):")
                print(correct_class_count_regular)
                print("Prototype-Based Model Correct Classifications (per class):")
                print(correct_class_count_proto)
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

    # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]
            # 平滑

        return protos

    def update_global_protos(
        self, new_local_protos, global_protos, glprotos_invol_dataset
    ):
        if self.client_protos and global_protos is not None:
            if self.args.agg_type == 3:
                agg_protos_label = defaultdict(list)
                all_labels = set()

                for [label, proto] in new_local_protos.items():
                    all_labels.add(label)

                len_sums = {k: 0 for k in all_labels}
                agg_protos_label = {
                    k: torch.zeros(self.args.feature_dim) for k in all_labels
                }

                for label in all_labels:
                    len_sums[label] = glprotos_invol_dataset[label]
                    if global_protos is not None and label in global_protos.keys():
                        global_protos[label] = global_protos[label] * len_sums[label]

                    if label in new_local_protos and not torch.all(
                        new_local_protos[label].eq(0)
                    ):
                        if self.client_protos is not None:
                            global_protos[label] -= (
                                self.client_protos[label] * self.label_counts[label]
                            )

                        protoMultiDatalen = (
                            new_local_protos[label] * self.label_counts[label]
                        )

                        agg_protos_label[label] = torch.add(
                            agg_protos_label[label].to(self.device),
                            protoMultiDatalen.to(self.device),
                        )
                        assert len(agg_protos_label[label]) == self.args.feature_dim

                for label in all_labels:
                    if global_protos is not None and label in global_protos.keys():
                        agg_protos_label[label] += global_protos[label]
                    agg_protos_label[label] /= len_sums[label]

                if global_protos is not None:
                    for k in global_protos.keys():
                        if k not in agg_protos_label.keys():
                            agg_protos_label[k] = global_protos[k]
            return agg_protos_label
        else:
            return global_protos


def add_gaussian_noise_to_prototypes(protos, sigma=0.05):
    """
    Add Gaussian noise to prototypes for learning robustness.

    Args:
        protos (dict): Current global prototypes.
        sigma (float): Standard deviation of Gaussian noise.

    Returns:
        dict: Prototypes with added Gaussian noise.
    """
    for label in protos.keys():
        noise = torch.normal(mean=0, std=sigma, size=protos[label].shape).to(
            protos[label].device
        )
        protos[label] += noise
    return protos


# def prototype_distance_loss(rep, global_protos, y, margin=100):
#     """
#     增加当前样本的中间向量 rep 与其他类别原型之间的距离。

#     参数:
#         rep (torch.Tensor): 当前样本的中间向量。
#         global_protos (dict): 全部类别的原型。
#         y (torch.Tensor): 当前样本的标签。
#         margin (float): 与其他类别原型的最小距离。增大此值会增加距离。

#     返回:
#         torch.Tensor: 距离损失。
#     """
#     loss = 0.0
#     for i, yy in enumerate(y):
#         y_c = yy.item()  # 当前样本的类别
#         current_proto = global_protos[y_c]  # 当前类别的原型

#         # 当前样本与当前类别原型的距离
#         rep_to_proto_distance = torch.norm(rep[i] - current_proto)

#         # 增加与其他类别原型的距离
#         for cls, proto in global_protos.items():
#             if cls != y_c:  # 不包括当前类别
#                 # 增加当前样本与其他类别原型的距离
#                 loss += torch.max(torch.norm(rep[i] - proto) - rep_to_proto_distance + margin, torch.tensor(0.0))

#     return loss


def orthogonal_constraint_loss(global_protos):
    """
    计算原型之间的正交约束损失，以增强不同类原型之间的差异。

    参数:
        global_protos (dict): 每个类别的原型。

    返回:
        torch.Tensor: 正交约束损失。
    """
    loss = 0.0
    proto_list = list(global_protos.values())
    num_protos = len(proto_list)

    # 计算每一对原型之间的余弦相似度
    for i in range(num_protos):
        for j in range(i + 1, num_protos):
            cosine_similarity = torch.dot(proto_list[i], proto_list[j]) / (
                proto_list[i].norm() * proto_list[j].norm()
            )
            loss += torch.abs(cosine_similarity)

    # 返回平均损失
    return loss / (num_protos * (num_protos - 1) / 2)


def distillation_loss(local_output, global_output, temperature=2.0):
    """
    local_output: 局部分类器的 logits 输出
    global_output: 全局分类器的 logits 输出
    temperature: 温度参数，用于调整概率分布平滑性
    """
    # 转为概率分布
    local_probs = F.softmax(local_output / temperature, dim=-1)
    global_probs = F.softmax(global_output / temperature, dim=-1)

    # KL散度损失
    loss = F.kl_div(local_probs.log(), global_probs, reduction="batchmean") * (
        temperature**2
    )
    return loss
