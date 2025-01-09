import os
import torch

def load_item(role, item_name, save_folder_name):
    file_path = os.path.join(save_folder_name, f"{role}_{item_name}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    return None

def save_item(item, role, item_name, save_folder_name):
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    torch.save(item, os.path.join(save_folder_name, f"{role}_{item_name}.pt")) 