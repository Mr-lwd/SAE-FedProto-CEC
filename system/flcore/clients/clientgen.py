import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.io_utils import load_item, save_item

class clientGen(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        trainloader = self.load_train_data()
        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        self.qualified_labels = []
        

    def train(self):
        head = load_item("Server", "head", self.save_folder_name)
        if head is not None:
            self.set_parameters()
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        generative_model = load_item('Server', 'generative_model', self.save_folder_name)
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
                # output = model(x)
                rep = model.base(x)
                rep = rep.squeeze(1)
                output = model.head(rep)
                loss = self.loss(output, y)
                self.local_model_loss += loss.item()
                if generative_model is not None:
                    labels = np.random.choice(self.qualified_labels, self.batch_size)
                    labels = torch.LongTensor(labels).to(self.device)
                    z = generative_model(labels)
                    loss += self.loss(model.head(z), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        self.local_model_loss = self.local_model_loss / len(trainloader)
        # self.local_all_loss = self.local_all_loss / len(trainloader)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
            
        
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name)
        head = load_item('Server', 'head', self.save_folder_name)
        for new_param, old_param in zip(head.parameters(), model.head.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, 'model', self.save_folder_name)

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        generative_model = load_item('Server', 'generative_model', self.save_folder_name)
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
                
                if generative_model is not None:
                    labels = np.random.choice(self.qualified_labels, self.batch_size)
                    labels = torch.LongTensor(labels).to(self.device)
                    z = generative_model(labels)
                    loss += self.loss(model.head(z), labels)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
