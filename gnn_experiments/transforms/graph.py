import math
import torch
from torch_geometric.data import Data

class PretrainingAttr(object):
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

        data.masked_node_indices = subset_mask
        data.masked_node_label = data.x[subset_mask]
        # data.x = data.x.clone()
        data.x[subset_mask] = self.num_node_types

        return data

class TrainingAttr(object):
    def __init__(self, task, y_transform=None):
        self.task = task
        self.y_transform = y_transform

    def __call__(self, data):
        data, protein_dict = data
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        new_data.y = self.task.target(protein_dict)
        if self.task.task_type == 'regression':
            new_data.y = torch.tensor(new_data.y).view(-1, 1)
            if self.y_transform is not None:
                new_data.y = torch.from_numpy(self.y_transform.transform(
                    new_data.y).astype('float32'))
        if isinstance(self.task, ps_tasks.LigandAffinityTask):
            fp_maccs = torch.tensor(protein_dict['protein']['fp_maccs']).view(1, -1)
            fp_morgan_r2 = torch.tensor(protein_dict['protein']['fp_morgan_r2']).view(1, -1)
            new_data.other_x = torch.cat((fp_maccs, fp_morgan_r2), dim=-1).float()
        return new_data


class PairTrainingAttr(object):
    def __init__(self, task, y_transform=None):
        self.task = task
        self.y_transform = y_transform

    def __call__(self, data):
        data, protein_dict = data
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        return new_data
