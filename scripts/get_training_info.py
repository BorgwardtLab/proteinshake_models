"""
Takes a proteinshake dataset and returns:

    * deterministic splits
    * task type
    * evaluator function
    * sets the .y attribute
"""
import os
import os.path as osp
import json

from sklearn import metrics
from sklearn.model_selection import train_test_split

def compute_splits(dataset,
                   random_state=42,
                   train_ratio=.75,
                   validation_ratio=.15,
                   test_ratio=.10):

    _, size = dataset.proteins()
    inds = range(size)
    train, test = train_test_split(inds, test_size=1 - train_ratio)
    val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio))

    return train, val, test

def tokenize(dataset, target, level, label_process):
    proteins, _ = dataset.proteins()
    labels = {label_process(p[level][target]) for p in proteins}
    return {label: i for i, label  in enumerate(labels)}

def get_task_info(dataset):
    train, val, test = compute_splits(dataset)
    data_name = dataset.__class__.__name__

    label_processors = {
                        'EnzymeCommissionDataset': lambda label: label.split(".")[0],
                        'TMScoreDataset': lambda label: label
                        }

    cache_dir = '.proteinshake'
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)

    try:
        with open(osp.join(cache_dir, f"{data_name}.json"), "r") as t:
            task_dict = json.load(t)
    except FileNotFoundError:
        print(">>> Cached task info not found, computing it.")


        if data_name == 'EnzymeCommissionDataset':
            task_type = 'multi-class'
            target = 'EC'
            level = 'protein'
            token_map = tokenize(dataset, target, level, label_processors[data_name])
            num_labels = len(token_map)
        elif data_name == 'TMScoreDataset':
            task_type = 'regression'
            target = 'tm_score'
            num_labels = 1
            pass
        else:
            raise ValueError

        task_dict = {'task_type': task_type,
                    'target': target,
                    'token_map': token_map,
                    'num_classes': num_labels,
                    'train_ind': train,
                    'test_ind': test,
                    'val_ind': val
                    }

        with open(osp.join(cache_dir, f"{data_name}.json"), "w") as t:
            json.dump(task_dict, t)

    task_dict['label_processor'] = label_processors[data_name]
    return task_dict

def set_y_pyg(pyg_dataset, ps_dataset):
    task_dict = get_task_info(ps_dataset)
    for p in pyg_dataset:
        label_processor = task_dict['label_processor']
        target = task_dict['target']
        p.y = task_dict['token_map'][label_processor(getattr(p, target)[0])]

def get_evaluator(ps_dataset):
    """ Returns scikit metric for dataset.
    """
    task_dict = get_task_info(ps_dataset)
    if task_dict['task_type'] == 'multi-class':
        return metrics.precision_score
    if task_dict['task_type'] == 'regression':
        return metrics.mean_squared_error

def get_splits(ps_dataset):
    task_dict = get_task_info(ps_dataset)
    return {'train': task_dict['train_ind'],
            'val': task_dict['val_ind'],
            'test': task_dict['test_ind']
            }

if __name__ == "__main__":
    import torch
    from torch_geometric.data import DataLoader
    from torch_geometric.nn import GraphConv
    from proteinshake.datasets import EnzymeCommissionDataset

    d = EnzymeCommissionDataset()
    d_pyg = d.to_graph(eps=3).pyg()

    task_dict = get_task_info(d)

    set_y_pyg(d_pyg, d)
    evaluator = get_evaluator(d)
    splits = get_splits(d)
