import torch

class VoxelMaskingTransform():

    def __call__(self, args, mask_ratio=0.15):
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

class VoxelLigandAffinityTransform():

    def __init__(self, task):
        self.task = task

    def __call__(self, args):
        data, protein_dict = args
        return data, torch.tensor(self.task.target(protein_dict)).float()

class VoxelEnzymeClassTransform():

    def __init__(self, task):
        self.task = task

    def __call__(self, args):
        data, protein_dict = args
        return data, torch.eye(self.task.num_classes)[self.task.target(protein_dict)].float()
