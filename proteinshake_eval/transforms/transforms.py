from .graph import GraphTrainTransform
from .point import PointTrainTransform


def get_train_transform(representation, task, y_transform=None, max_len=None):
    if representation == 'graph':
        return GraphTrainTransform(task, y_transform)
    elif representation == 'point_cloud':
        return PointTrainTransform(task, y_transform, max_len=max_len)
    else:
        raise ValueError("Not implemented!")


def get_transformed_dataset(cfg, dataset, task, y_transform=None, max_len=None):
    if cfg.name == 'graph':
        data_transform = GraphTrainTransform(task, y_transform)
        return dataset.to_graph(eps=cfg.graph_eps).pyg(transform=data_transform)
    elif cfg == 'point_cloud':
        data_transform = PointTrainTransform(task, y_transform, max_len=max_len)
        dset.to_point().torch(transform=data_transform)
    else:
        raise ValueError("Not implemented!")
