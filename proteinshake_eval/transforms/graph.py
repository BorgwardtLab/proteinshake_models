import math
import torch
from torch_geometric.data import Data
from .utils import reshape_data, add_other_data


class GraphPretrainTransform(object):
    def __call__(self, data):
        data, _ = data
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)#data.residue_number - 1
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        return new_data

class MaskNode(object):
    def __init__(self, num_node_types, mask_rate=0.15):
        self.num_node_types = num_node_types
        self.mask_rate = mask_rate

    def __call__(self, data):
        num_nodes = data.num_nodes
        subset_mask = torch.rand(num_nodes) < self.mask_rate

        data.masked_indices = subset_mask
        data.masked_label = data.x[subset_mask]

        data.x[subset_mask] = self.num_node_types

        return data

class GraphTrainTransform(object):
    def __init__(self, task, y_transform=None):
        self.task = task
        _,self.task_type = task.task_type
        self.y_transform = y_transform

    def __call__(self, data):
        data, protein_dict = data
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        new_data.y = self.task.target(protein_dict)
        new_data = reshape_data(new_data, self.task_type)
        new_data = add_other_data(new_data, self.task, protein_dict)
        return new_data


class GraphPairTrainTransform(object):
    def __call__(self, data):
        data, protein_dict = data
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        return new_data
