# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients，(暂时只考虑一轮，忽略)
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

from collections import defaultdict
import copy
import random
from flcore.edges.edgebase import Edge
from flcore.clients.clientbase import load_item, save_item

import torch
import numpy as np


class Edge_FedTest(Edge):
    def __init__(self, args, id, cids):
        # 调用父类的构造函数
        super().__init__(args, id, cids, shared_layers=None)

