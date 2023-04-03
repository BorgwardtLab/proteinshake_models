import torch
from torch_geometric.data import Data
from .utils import reshape_data, add_other_data


class VoxelRotationAugment():

    def __call__(self, args):
        data, protein_dict = args
        rnd = int(torch.randint(0,2+1,(1,)))
        rot = int(torch.randint(1,3+1,(1,)))
        rotation_plane = {0:[0,1],1:[1,2],2:[0,2]}[rnd]
        data = torch.rot90(data,k=rot,dims=rotation_plane)
        return data, protein_dict

class VoxelMaskingTransform():
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio

    def __call__(self, args):
        mask_ratio = self.mask_ratio
        data, protein_dict = args
        nonzero = ~((data == 0).all(-1))
        volume = nonzero.sum()
        n, m = int(volume * mask_ratio), volume-int(volume * mask_ratio)
        mask = torch.zeros(data.shape[:-1]).bool()
        inner_mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(volume)].bool()
        mask[nonzero] = inner_mask
        masked = data.clone()
        masked[mask] = 1
        return data, masked, mask


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

# class VoxelLigandAffinityTransform():

#     def __init__(self, task, y_transform=None):
#         self.task = task
#         self.y_transform = y_transform

#     def __call__(self, args):
#         data, protein_dict = args
#         target = torch.tensor(self.task.target(protein_dict)).float()
#         if self.y_transform is not None:
#             target = torch.from_numpy(self.y_transform.transform(target).astype('float32'))
#         fp_maccs = torch.tensor(protein_dict['protein']['fp_maccs'])
#         fp_morgan_r2 = torch.tensor(protein_dict['protein']['fp_morgan_r2'])
#         fingerprint = torch.cat((fp_maccs, fp_morgan_r2), dim=-1).float()
#         return data, target, fingerprint

# class VoxelEnzymeClassTransform():

#     def __init__(self, task, y_transform=None):
#         self.task = task

#     def __call__(self, args):
#         data, protein_dict = args
#         return data, torch.eye(self.task.num_classes)[self.task.target(protein_dict)].float()

# class VoxelScopTransform():

#     def __init__(self, task, y_transform=None):
#         self.task = task

#     def __call__(self, args):
#         data, protein_dict = args
#         return data, torch.eye(self.task.num_classes)[self.task.target(protein_dict)].float()
