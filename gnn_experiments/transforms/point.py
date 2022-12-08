import torch
import torch.nn.functional as F
from proteinshake import tasks as ps_tasks
from torch_geometric.data import Data


class PointMaskingTransform():

    def __call__(self, data, protein_dict, mask_ratio=0.15):
        coords, labels = data[:,:3], data[:,3]
        labels = torch.eye(20)[labels.long()].float()
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        length = labels.shape[0]
        n, m = int(length * mask_ratio), length-int(length * mask_ratio)
        mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(length)].bool()
        masked = labels.clone()
        masked[mask] = torch.ones(labels.shape[1]).float()
        return coords, labels, masked, mask


class PointTrainingTransform(object):
    def __init__(self, task, y_transform=None, max_len=1000):
        self.task = task
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
        if 'binary' in self.task.task_type:
            data.y = torch.tensor(data.y).view(-1, 1).float()
        if self.task.task_type == 'regression':
            data.y = torch.tensor(data.y).view(-1, 1).float()
            if self.y_transform is not None:
                data.y = torch.from_numpy(self.y_transform.transform(
                    data.y).astype('float32'))
        if isinstance(self.task, ps_tasks.LigandAffinityTask):
            fp_maccs = torch.tensor(protein_dict['protein']['fp_maccs'])
            fp_morgan_r2 = torch.tensor(protein_dict['protein']['fp_morgan_r2'])
            other_x = torch.cat((fp_maccs, fp_morgan_r2), dim=-1).float()
            # return (coords.T, labels, mask), other_x, target
            data.other_x = other_x.view(1, -1)
        return data


class PointEnzymeClassTransform():

    def __init__(self, task):
        self.task = task

    def __call__(self, data):
        data, protein_dict = data
        coords, labels = data[:,:3], data[:,3]
        labels = torch.eye(20)[labels.long()].float()
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        ec = torch.eye(self.task.num_classes)[self.task.target(protein_dict)].float()
        return coords, labels, ec


class PointLigandAffinityTransform():

    def __init__(self, task):
        self.task = task

    def transform(self, data, protein_dict):
        coords, labels = data[:,:3], data[:,3]
        labels = torch.eye(20)[labels.long()].float()
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        la = torch.tensor(self.task.target(protein_dict)).float()
        return coords, labels, la
