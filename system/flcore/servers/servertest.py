import time
import numpy as np

from flcore.clients.clienttest import clientTest
from flcore.edges.edgetest import Edge_FedTest
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.func_utils import *
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import copy
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ServerTest(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.glprotos_invol_dataset = defaultdict(int)

        # select slow clients
        # self.set_slow_clients()
        # 初始化所有客户端
        self.set_clients(clientTest)
        # 初始化所有边缘服务器
        # self.set_edges(Edge_FedTest)

        self.compute_glprotos_invol_dataset()
        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("No creating server and clients.")


