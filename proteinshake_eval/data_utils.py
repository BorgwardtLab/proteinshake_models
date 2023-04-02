import torch
import numpy as np


class PPIDataset(object):
    splits = ['train', 'val', 'test']
    def __init__(self, dataset, task, split='train', filter_mask=None,
                 transform=None, y_transform=None):
        self.dataset = dataset
        self.task = task
        self.transform = transform
        self.y_transform = y_transform

        assert split in self.splits, f"split should be in {self.splits}"
        self.indices = getattr(task, '{}_ind'.format(split))
        if filter_mask is not None:
            self.indices = np.asarray(self.indices)[filter_mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, j = self.indices[index]
        data1, protein_dict1 = self.dataset[i]
        data2, protein_dict2 = self.dataset[j]
        y = torch.tensor(self.task.target(protein_dict1, protein_dict2))
        if self.task.task_type == 'regression':
            y = y.view(1)
        if self.y_transform is not None:
            y = torch.from_numpy(self.y_transform.transform(y.view(1, -1)).astype('float32')).view(1)

        if self.transform is not None:
            data1 = self.transform((data1, protein_dict1))
            data2 = self.transform((data2, protein_dict2))

        return data1, data2, y
