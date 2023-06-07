import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from .utils import reshape_data, add_other_data


class PointTrainTransform(object):
    def __init__(self, task, y_transform=None, max_len=1000):
        self.task = task
        _,self.task_type = task.task_type
        self.y_transform = y_transform
        self.max_len = max_len

    def __call__(self, data):
        data, protein_dict = data
        coords, labels = data[:,:3], data[:,3]
        data = Data()
        # labels = torch.eye(20)[labels.long()].float()
        mask = torch.zeros((self.max_len,), dtype=torch.bool)
        mask[:min(coords.shape[0], self.max_len)] = True
        coords = F.pad(coords[:self.max_len], (0,0,0,max(0, self.max_len - coords.shape[0])))
        # labels = F.pad(labels[:self.max_len], (0,0,0,max(0, self.max_len - labels.shape[0])))
        labels = F.pad(labels[:self.max_len].long(), (0, max(0, self.max_len - labels.shape[0])), value=21)
        target = self.task.target(protein_dict)
        data.coords = coords.T.unsqueeze(0)
        data.labels = labels.unsqueeze(0)
        data.mask = mask.unsqueeze(0)
        data.y = target
        data = reshape_data(data, self.task_type)
        data = add_other_data(data, self.task, protein_dict)
        return data


class PointPairTrainTransform(object):
    def __init__(self, max_len=1000):
        self.max_len = max_len

    def __call__(self, data):
        data, protein_dict = data
        coords, labels = data[:,:3], data[:,3]
        data = Data()
        mask = torch.zeros((self.max_len,), dtype=torch.bool)
        mask[:min(coords.shape[0], self.max_len)] = True
        coords = F.pad(coords[:self.max_len], (0,0,0,max(0, self.max_len - coords.shape[0])))
        labels = F.pad(labels[:self.max_len].long(), (0, max(0, self.max_len - labels.shape[0])), value=21)
        data.coords = coords.T.unsqueeze(0)
        data.labels = labels.unsqueeze(0)
        data.mask = mask.unsqueeze(0)
        return data


class PointPretrainTransform(object):
    def __init__(self, max_len=1000):
        self.max_len = max_len

    def __call__(self, data):
        data, _ = data
        coords, labels = data[:,:3], data[:,3]
        data = Data()
        mask = torch.zeros((self.max_len,), dtype=torch.bool)
        num_points = min(coords.shape[0], self.max_len)
        mask[:num_points] = True
        coords = F.pad(coords[:self.max_len], (0,0,0,max(0, self.max_len - coords.shape[0])))
        labels = F.pad(labels[:self.max_len].long(), (0, max(0, self.max_len - labels.shape[0])), value=21)
        data.num_points = num_points
        data.coords = coords.T.unsqueeze(0)
        data.labels = labels.unsqueeze(0)
        data.mask = mask.unsqueeze(0)
        return data

class MaskPoint(object):
    def __init__(self, num_point_types=20, mask_rate=0.15):
        self.num_point_types = num_point_types
        self.mask_rate = mask_rate

    def __call__(self, data):
        num_points = data.num_points
        max_len = data.coords.shape[-1]
        subset_mask = torch.rand(num_points) < self.mask_rate
        subset_mask = F.pad(subset_mask, (0, max_len - num_points))

        data.masked_indices = subset_mask.unsqueeze(0)
        data.masked_label = data.labels[0, subset_mask]
        data.labels[0, subset_mask] = self.num_point_types

        return data
