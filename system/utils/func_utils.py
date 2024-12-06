import torch
import math


def default_tensor(feature_dim=512):
    return torch.zeros(feature_dim)


def get_transmission_time(M=0):
    # 定义所有的参数, hierachical论文设置
    B = 1 * 10**6  # 信道带宽，1 MHz = 10^6 Hz
    h = 10**-8  # 信道增益
    p = 0.5  # 发射功率，0.5 W
    sigma = 10**-10  # 噪声功率，10^-10 W
    division = 51  # 1 + (h * p) / sigma = 51
    # 计算通信延迟
    T_comm = M / (B * math.log2(division))  # 计算通信延迟
    E_comm = p * T_comm  # 计算通信能耗

    return T_comm, E_comm


def get_theory_bytes(protos={}):
    all_bytes = 0
    for key, tensor_list in protos.items():
        if isinstance(tensor_list, torch.Tensor):
            tensor_size_bytes = tensor_list.element_size() * tensor_list.nelement()
            all_bytes += tensor_size_bytes

        elif hasattr(tensor_list, '__iter__'):
            for tensor in tensor_list:
                if isinstance(tensor, torch.Tensor):
                    tensor_size_bytes = tensor.element_size() * tensor.nelement()
                    all_bytes += tensor_size_bytes
        else:
            print(f"Key {key}: tensor_list is neither a tensor nor a list.")

    return all_bytes
