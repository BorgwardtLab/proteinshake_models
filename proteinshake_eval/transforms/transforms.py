from .graph import GraphTrainTransform, GraphPairTrainTransform
from .point import PointTrainTransform, PointPairTrainTransform
from .voxel import VoxelTrainTransform, VoxelPairTrainTransform
from .utils import PPIDataset


# def get_train_transform(representation, task, y_transform=None, max_len=None):
#     if representation == 'graph':
#         return GraphTrainTransform(task, y_transform)
#     elif representation == 'point_cloud':
#         return PointTrainTransform(task, y_transform, max_len=max_len)
#     else:
#         raise ValueError("Not implemented!")


def get_transformed_dataset(cfg, dataset, task, y_transform=None, max_len=None):
    if task.task_type[0] == "protein_pair":
        if cfg.name == 'graph':
            data_transform = GraphPairTrainTransform()
            dataset = dataset.to_graph(eps=cfg.graph_eps).pyg()
        elif cfg.name == 'point_cloud':
            data_transform = PointPairTrainTransform(max_len=max_len)
            dataset = dataset.to_point().torch()
        elif cfg.name == 'voxel':
            data_transform = VoxelPairTrainTransform()
            dataset = dataset.to_voxel(
                gridsize=cfg.gridsize, voxelsize=cfg.voxelsize).torch()
        else:
            raise ValueError("Not implemented!")

        train_dset = PPIDataset(dataset, task, 'train', transform=data_transform, y_transform=y_transform)
        val_dset = PPIDataset(dataset, task, 'val', transform=data_transform, y_transform=y_transform)
        test_dset = PPIDataset(dataset, task, 'test', transform=data_transform, y_transform=y_transform)
        return train_dset, val_dset, test_dset

    if cfg.name == 'graph':
        data_transform = GraphTrainTransform(task, y_transform)
        return dataset.to_graph(eps=cfg.graph_eps).pyg(transform=data_transform)
    elif cfg.name == 'point_cloud':
        data_transform = PointTrainTransform(task, y_transform, max_len=max_len)
        return dataset.to_point().torch(transform=data_transform)
    elif cfg.name == 'voxel':
        data_transform = VoxelTrainTransform(task, y_transform)
        return dataset.to_voxel(
            gridsize=cfg.gridsize, voxelsize=cfg.voxelsize).torch(transform=data_transform)
    else:
        raise ValueError("Not implemented!")
