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

    def __init__(self, task, y_transform=None):
        self.task = task
        self.y_transform = y_transform

    def __call__(self, args):
        data, protein_dict = args
        target = torch.tensor(self.task.target(protein_dict)).float()
        if self.y_transform is not None:
            target = torch.from_numpy(self.y_transform.transform(target).astype('float32'))
        fp_maccs = torch.tensor(protein_dict['protein']['fp_maccs'])
        fp_morgan_r2 = torch.tensor(protein_dict['protein']['fp_morgan_r2'])
        fingerprint = torch.cat((fp_maccs, fp_morgan_r2), dim=-1).float()
        return data, target, fingerprint

class VoxelEnzymeClassTransform():

    def __init__(self, task, y_transform=None):
        self.task = task

    def __call__(self, args):
        data, protein_dict = args
        return data, torch.eye(self.task.num_classes)[self.task.target(protein_dict)].float()
