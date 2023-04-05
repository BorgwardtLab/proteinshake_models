import math
import torch
from torch_geometric.data import Data
from torch import nn
import numpy as np
import importlib

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


def get_task(task_name):
    all_task_classes = importlib.import_module('proteinshake.tasks')
    return getattr(all_task_classes, task_name)


# class OneHotToIndex(object):
#     def __call__(self, data):
#         data.x = data.x.argmax(dim=-1)
#         return data

# class ResidueIdx(object):
#     def __call__(self, data):
#         data.residue_idx = torch.arange(data.num_nodes)
#         return data

def get_filter_mask(dataset, task, n=3000):
    """Filter proteins with length > n
    """
    protein_len_list = np.asarray(
        [len(protein_dict['protein']['sequence']) for protein_dict in dataset.proteins()])
    percent = np.sum(protein_len_list <= n) / len(protein_len_list) * 100
    print(f"Protein length less or equal to {n} is {percent}%")
    train_mask = protein_len_list[task.train_index] <= n
    val_mask = protein_len_list[task.val_index] <= n
    test_mask = protein_len_list[task.test_index] <= n
    return train_mask, val_mask, test_mask

def get_data_loaders(dataset, task, masks, batch_size, num_workers):
    if task.task_type[0] == "protein_pair":
        train_dset, val_dset, test_dset = dataset
        train_loader = DataLoader(train_dset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader
    train_mask, val_mask, test_mask = masks
    train_loader = DataLoader(
        Subset(dataset, np.asarray(task.train_index)[train_mask]),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        Subset(dataset, np.asarray(task.val_index)[val_mask]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        Subset(dataset, np.asarray(task.test_index)[test_mask]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_loss(task_type):
    if task_type == "multi_class" or task_type == 'multi-class':
        return nn.CrossEntropyLoss(), 'accuracy'
    elif task_type == "multi_label":
        return nn.BCEWithLogitsLoss(), 'Fmax'
    elif task_type == "binary":
        return nn.BCEWithLogitsLoss(), 'accuracy'
    elif task_type == "regression":
        return nn.L1Loss(), 'spearmanr'

# class Aggregator(nn.Module):
#     def __init__(self, embed_dim=256, aggregation='concat', normalize=False):
#         super().__init__()
#         self.aggregation = aggregation
#         self.normalize = normalize

#         if aggregation == 'concat':
#             self.aggregator = nn.Sequential(
#                 nn.Linear(2 * embed_dim, embed_dim),
#                 nn.ReLU(True),
#                 nn.Linear(embed_dim, embed_dim)
#             )
#         elif aggregation == 'dot' or aggregation == 'sum':
#             self.aggregator = nn.Sequential(
#                 nn.Linear(embed_dim, embed_dim),
#                 nn.ReLU(True),
#                 nn.Linear(embed_dim, embed_dim)
#             )

#     def forward(self, x1, x2):
#         if self.normalize:
#             x1 = F.normalize(x1, dim=-1)
#             x2 = F.normalize(x2, dim=-1)
#         if self.aggregation == 'concat':
#             x = torch.cat((x1, x2), dim=-1)
#         elif self.aggregation == 'dot':
#             x = x1 * x2
#         elif self.aggregation == 'sum':
#             x = x1 + x2
#         return self.aggregator(x)
