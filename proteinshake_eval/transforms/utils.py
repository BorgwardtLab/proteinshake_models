import torch
import numpy as np
import proteinshake.tasks as ps_tasks


def reshape_data(data, task_type, y_transform=None):
    if 'binary' in task_type:
        data.y = torch.tensor(data.y).view(-1, 1).float()
    if task_type == 'multi_label':
        data.y = torch.tensor(data.y).view(1, -1).float()
    if task_type == 'regression':
        data.y = torch.tensor(data.y).view(-1, 1).float()
        if y_transform is not None:
            data.y = torch.from_numpy(y_transform.transform(data.y).astype('float32'))
    return data

def add_other_data(data, task, protein_dict):
    if isinstance(task, ps_tasks.LigandAffinityTask):
        fp_maccs = torch.tensor(protein_dict['protein']['fp_maccs'])
        fp_morgan_r2 = torch.tensor(protein_dict['protein']['fp_morgan_r2'])
        other_x = torch.cat((fp_maccs, fp_morgan_r2), dim=-1).float()
        data.other_x = other_x.view(1, -1)
    return data

class PPIDataset(object):
    splits = ['train', 'val', 'test']
    def __init__(self, dataset, task, split='train', filter_mask=None,
                 transform=None, y_transform=None):
        self.dataset = dataset
        self.task = task
        _,self.task_type = task.task_type
        self.transform = transform
        self.y_transform = y_transform

        self.set_split(split, filter_mask)

    def set_split(self, split='train', filter_mask=None):
        self.split = split
        assert split in self.splits, f"split should be in {self.splits}"
        self.indices = getattr(self.task, '{}_index'.format(split))
        if filter_mask is not None:
            self.indices = np.asarray(self.indices)[filter_mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, j = self.indices[index]
        data1, protein_dict1 = self.dataset[i]
        data2, protein_dict2 = self.dataset[j]
        if self.transform is not None:
            data1 = self.transform((data1, protein_dict1))
            data2 = self.transform((data2, protein_dict2))

        y = torch.tensor(self.task.target(protein_dict1, protein_dict2)).float()
        s = None
        if self.task_type == 'binary':
            from torch_geometric.data import Data
            if self.split == 'train':
                pos_idx = y.nonzero().T
                pos_size = pos_idx.shape[1]
                neg_size = pos_size# * 3
                neg_idx = torch.vstack((
                    torch.randint(data1.num_nodes, (neg_size,)),
                    torch.randint(data2.num_nodes, (neg_size,))
                ))
                s = torch.hstack((pos_idx, neg_idx))
                y = y[s[0], s[1]]
            else:
                s = torch.vstack((
                    torch.arange(data1.num_nodes).repeat_interleave(data2.num_nodes),
                    torch.arange(data2.num_nodes).repeat(data1.num_nodes)
                ))
            y = Data(y=y.view(-1, 1))
        if self.task_type == 'regression':
            y = y.view(1)
        if self.y_transform is not None:
            y = torch.from_numpy(self.y_transform.transform(y.view(1, -1)).astype('float32')).view(1)

        if s is not None:
            data1.sub_index = s[0]
            data2.sub_index = s[1]

        return data1, data2, y
