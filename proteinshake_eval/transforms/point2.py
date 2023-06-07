import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from .utils import reshape_data, add_other_data
import torch_geometric.transforms as T

transform = T.Compose([
    T.NormalizeScale(),
])

pretrain_transform = T.Compose([
    T.NormalizeScale(),
    T.RandomJitter(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])


class PointTrainTransform(object):
    def __init__(self, task, y_transform=None):
        self.task = task
        _,self.task_type = task.task_type
        self.y_transform = y_transform

    def __call__(self, data):
        data, protein_dict = data
        pos, x = data[:,:3], data[:,3]
        data = Data()

        target = self.task.target(protein_dict)
        data.pos = pos
        data.x = x.long()
        data.y = target
        data = reshape_data(data, self.task_type)
        data = add_other_data(data, self.task, protein_dict)
        return transform(data)


class PointPairTrainTransform(object):

    def __call__(self, data):
        data, protein_dict = data
        pos, x = data[:,:3], data[:,3]
        data = Data()
        data.pos = pos
        data.x = x.long()
        return transform(data)


class PointPretrainTransform(object):
    def __call__(self, data):
        data, _ = data
        pos, x = data[:,:3], data[:,3]
        data = Data()
        data.pos = pos
        data.x = x.long()
        return pretrain_transform(data)

class MaskPoint(object):
    def __init__(self, num_point_types=20, mask_rate=0.15):
        self.num_point_types = num_point_types
        self.mask_rate = mask_rate

    def __call__(self, data):
        num_points = data.num_nodes
        subset_mask = torch.rand(num_points) < self.mask_rate

        data.masked_indices = subset_mask
        data.masked_label = data.x[subset_mask]

        data.x[subset_mask] = self.num_point_types

        return data
