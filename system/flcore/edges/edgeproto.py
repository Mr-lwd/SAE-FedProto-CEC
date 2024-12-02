# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients，(暂时只考虑一轮，忽略)
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from flcore.edges.edgebase import Edge
import numpy as np


class Edge_FedProto(Edge):
    def __init__(self, args, id, cids):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_protos: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_protos: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param eshared_protos: SharedProtos store in edge server
        :return:
        """
        # 调用父类的构造函数
        super().__init__(args, id, cids, shared_layers=None)
        self.eshared_protos_global = None
        self.eshared_protos_local = None


    def train(self, clients):
        print(f"Edge {self.id} begin training")
        self.refresh_edgeserver()
        selected_cnum = max(int(self.clients_per_edge * self.args.join_ratio), 1)
        self.join_clients = selected_cnum # 记录本轮参与训练的客户端数量
        self.selected_cids = np.random.choice(
            self.cids, selected_cnum, replace=False, p=self.p_clients
        )
        for selected_cid in self.selected_cids:
            self.client_register(clients[selected_cid])
        for selected_cid in self.selected_cids:
            self.send_to_client(clients[selected_cid])
            clients[selected_cid].sync_with_edgeserver()
            clients[selected_cid].train()
            # clients[selected_cid].send_to_edgeserver(self)
        # edge_loss[i] = client_loss
        # edge_sample[i] = sum(self.sample_registration.values())
        # self.aggregate()
        
    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.id_registration.clear()
        # del self.id_registration[:]
        self.sample_registration.clear()

        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = client.train_samples
        return None

    def receive_from_client(self, client_id, cshared_protos_local):
        self.receiver_buffer[client_id] = cshared_protos_local
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        # sample_num = [snum for snum in self.sample_registration.values()]

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.eshared_protos_global))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(
            edge_id=self.id,
            eshared_protos_local=copy.deepcopy(self.eshared_protos_local),
        )
        return None

    def receive_from_cloudserver(self, cloud_shared_protos):
        self.eshared_protos_global = cloud_shared_protos
        return None
