import torch
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from utils.io_utils import load_item, save_item
from collections import defaultdict

class clientGH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

    def train(self):
        head = load_item("Server", "head", self.save_folder_name)
        if head is not None:
            self.set_parameters()
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        total_loss = 0
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
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

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            self.local_model_loss = 0
            self.local_all_loss = 0
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
                loss = self.loss(output, y)
                self.local_model_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(model, self.role, "model", self.save_folder_name)
        self.local_model_loss = self.local_model_loss / len(trainloader)
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def set_parameters(self):
        model = load_item(self.role, "model", self.save_folder_name)
        head = load_item("Server", "head", self.save_folder_name)
        for new_param, old_param in zip(head.parameters(), model.head.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, "model", self.save_folder_name)

    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                # rep = rep.squeeze(1)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        save_item(agg_func(protos), self.role, "protos", self.save_folder_name)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
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

    return protos
