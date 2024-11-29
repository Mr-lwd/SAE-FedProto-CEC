# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients，(暂时只考虑一轮，忽略)
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy


class Edge:

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
        :param shared_protos: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.role = "Edge_" + str(self.id)
        self.receiver_buffer = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.eshared_protos = None
        self.clock = []

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = client.train_samples
        return None

    def receive_from_client(self, client_id, cshared_protos):
        self.receiver_buffer[client_id] = cshared_protos
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        # sample_num = [snum for snum in self.sample_registration.values()]
        # self.shared_protos = average_weights(w = received_dict,
        #                                          s_num= sample_num)

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.eshared_protos))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(
            edge_id=self.id, eshared_protos=copy.deepcopy(self.eshared_protos)
        )
        return None

    def receive_from_cloudserver(self, global_shared_protos):
        self.eshared_protos = global_shared_protos
        return None
