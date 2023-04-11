import torch
from torch_geometric.data import Data
from .utils import reshape_data, add_other_data


class VoxelRotationAugment(object):

    def __call__(self, args):
        data, protein_dict = args
        rnd = int(torch.randint(0,2+1,(1,)))
        rot = int(torch.randint(1,3+1,(1,)))
        rotation_plane = {0:[0,1],1:[1,2],2:[0,2]}[rnd]
        data = torch.rot90(data,k=rot,dims=rotation_plane)
        return data, protein_dict

class VoxelPretrainTransform(object):
    def __init__(self, mask_rate=0.15):
        self.mask_rate = mask_rate

    def __call__(self, data):
        data, protein_dict = data
        nonzero = ~((data == 0).all(-1))
        volume = nonzero.sum()
        n, m = int(volume * self.mask_rate), volume - int(volume * self.mask_rate)
        mask = torch.zeros(data.shape[:-1]).bool()
        inner_mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(volume)].bool()
        mask[nonzero] = inner_mask
        masked = data.clone()
        masked[mask] = 1

        batch_data = Data()
        batch_data.x = masked.unsqueeze(0)
        batch_data.masked_indices = mask.unsqueeze(0)
        batch_data.masked_label = data[mask].argmax(-1)
        return batch_data


class VoxelTrainTransform(object):
    def __init__(self, task, y_transform=None, use_totation_aug=True):
        self.task = task
        _,self.task_type = task.task_type
        self.y_transform = y_transform
        self.augment = VoxelRotationAugment() if use_totation_aug else None

    def __call__(self, data):
        data, protein_dict = self.augment(data)
        batch_data = Data()
        batch_data.x = data.unsqueeze(0)
        batch_data.mask = ((data == 0).all(-1)).unsqueeze(0)
        batch_data.y = self.task.target(protein_dict)
        # reshape y
        batch_data = reshape_data(batch_data, self.task_type)
        # add other data if self.task is in [LigandAffinityTask, ]
        batch_data = add_other_data(batch_data, self.task, protein_dict)
        return batch_data


class VoxelPairTrainTransform(object):
    def __init__(self, use_totation_aug=True):
        self.augment = VoxelRotationAugment() if use_totation_aug else None

    def __call__(self, data):
        data, protein_dict = self.augment(data)
        batch_data = Data()
        batch_data.x = data.unsqueeze(0)
        batch_data.mask = ((data == 0).all(-1)).unsqueeze(0)
        return batch_data
