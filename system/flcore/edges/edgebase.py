# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

from collections import defaultdict
import copy
from flcore.edges.average import average_weights
from utils.io_utils import load_item, save_item
from utils.func_utils import *


class Edge:
    def __init__(self, args, id, cids, shared_layers=None):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.args = args
        self.id = id
        self.role = "Edge_" + str(self.id)
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.have_participated_ids =  set()
        self.sample_registration = {}
        self.all_trainsample_num = 0
        # self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.p_clients = []  # 每个边缘服务器上的客户端数据量比例
        self.selected_cids = []  # 每轮选择的客户端id
        self.clients_per_edge = int(args.num_clients / args.num_edges)
        self.join_clients = 0
        self.save_folder_name = args.save_folder_name_full
        self.device = args.device

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = client.train_samples
        return None

    # def client_register(self, client):
    #     self.id_registration.append(client.id)
    #     self.sample_registration[client.id] = len(client.train_loader.dataset)
    #     return None

    def receive_from_client(self, client_id, cshared_state_dict):
        self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(
            edge_id=self.id, eshared_state_dict=copy.deepcopy(self.shared_state_dict)
        )
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None

    def edgeAggregate(self, clients):
        # 直接从当前轮次参与的客户端id读取聚合后原型和{clientid: {label: [feature], ...}, ...} Set
        clientProtos = {
            id: load_item(clients[id].role, "protos", self.save_folder_name)
            for id in self.cids
        }

        clientProtos_prev = {
            id: load_item(clients[id].role, "prev_protos", self.save_folder_name)
            for id in self.cids
        }

        edgeProtos = defaultdict(default_tensor)
        for j in range(self.args.num_classes):
            for id in self.cids:
                if id in self.id_registration and j in clientProtos[id].keys():
                    edgeProtos[j] = edgeProtos[j].to(self.device)
                    edgeProtos[j] += (clients[id].label_counts[j]) * clientProtos[id][
                        j
                    ].squeeze()
                    if clientProtos_prev[id] is None:
                        self.N_l[j] += clients[id].label_counts[j]
                    assert len(edgeProtos[j]) == self.args.feature_dim
                elif (
                    clientProtos_prev[id] is not None
                    and j in clientProtos_prev[id].keys()
                ):
                    edgeProtos[j] = edgeProtos[j].to(self.device)
                    edgeProtos[j] += (
                        clients[id].label_counts[j] * clientProtos_prev[id][j].squeeze()
                    )
                    assert len(edgeProtos[j]) == self.args.feature_dim

            if self.N_l[j] != 0:
                edgeProtos[j] = edgeProtos[j] / self.N_l[j]  # 平均

        save_item(edgeProtos, self.role, "protos", self.save_folder_name)

        # for id in self.id_registration:
        #     if clientProtos[id] is not None:
        #         save_item(
        #             clientProtos[id],
        #             clients[id].role,
        #             "prev_protos",
        #             self.save_folder_name,
        #         )
