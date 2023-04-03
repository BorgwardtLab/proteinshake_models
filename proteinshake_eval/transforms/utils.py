import torch
import proteinshake.tasks as ps_tasks


def reshape_data(data, task_type, y_transform=None):
    if 'binary' in task_type:
            data.y = torch.tensor(data.y).view(-1, 1).float()
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
