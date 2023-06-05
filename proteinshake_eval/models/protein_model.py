import torch
from torch import nn
from .aggregator import Aggregator
from .graph import GNN_encoder
from .point import PointNet_encoder
from .point2 import PointNetPlusPlus_encoder
from .voxel import VoxelNet_encoder


NUM_PROTEINS = 20

def build_encoder(cfg):
    if cfg.name == 'gnn':
        return GNN_encoder(
            cfg.embed_dim,
            cfg.num_layers,
            cfg.dropout,
            cfg.gnn_type,
            cfg.use_edge_attr,
            cfg.pe,
            cfg.pooling,
        )
    elif cfg.name == 'point_net':
        # return PointNet_encoder(
        #     cfg.embed_dim,
        #     cfg.pooling,
        #     cfg.alpha
        # )
        return PointNetPlusPlus_encoder(
            cfg.embed_dim,
            cfg.pooling
        )
    elif cfg.name == 'voxel_net':
        return VoxelNet_encoder(
            cfg.embed_dim,
            cfg.num_layers,
            cfg.kernel_size,
            cfg.dropout,
            cfg.pooling,
        )
    else:
        raise ValueError("Not implemented!")

class TaskHead(nn.Module):
    def __init__(self, task, embed_dim=256, out_head='linear',
                 aggregation='dot'):
        super().__init__()

        task_level, task_type = task.task_type

        self.pair_prediction = task.pair_data

        if self.pair_prediction:
            if not 'pair' in task_level:
                other_dim = task.other_dim
                self.other_encoder = nn.Sequential(
                    nn.Linear(other_dim, embed_dim),
                    nn.BatchNorm1d(embed_dim),
                    nn.ReLU(True),
                    nn.Linear(embed_dim, embed_dim),
                    nn.BatchNorm1d(embed_dim),
                )
            self.aggregator = Aggregator(embed_dim, aggregation)

        num_class = task.num_classes if 'multi' in task_type else 1

        if out_head == 'linear':
            self.classifier = nn.Linear(embed_dim, num_class)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(True),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.ReLU(True),
                nn.Linear(embed_dim // 4, num_class)
            )

    def forward(self, x, other_x=None):
        if self.pair_prediction:
            assert other_x is not None, "other_data should be provided!"
            if hasattr(self, 'other_encoder'):
                other_x = self.other_encoder(other_x)
            x = self.aggregator(x, other_x)

        return self.classifier(x)


class ProteinStructureEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.encoder = build_encoder(cfg)

        self.head = nn.Linear(cfg.embed_dim, NUM_PROTEINS)

    def forward(self, data):
        output = self.encoder(data)
        return self.head(output[data.masked_indices])

    def step(self, batch):
        y = batch.masked_label
        y_hat = self(batch)
        return y_hat, y

    def save(self, path):
        torch.save(
            {'cfg': self.cfg,
            'state_dict': self.encoder.encoder.state_dict(),
            'head_state_dict': self.head.state_dict()
            },
            path
        )


class ProteinStructureNet(nn.Module):
    def __init__(self, cfg, task):
        super().__init__()
        self.cfg = cfg
        # Build protein encoder
        self.encoder = build_encoder(cfg)

        # Build task head
        self.task_head = self.build_task_head(cfg, task)

        self.encode_other = task.pair_data
        self.encode_other_protein = True if 'pair' in task.task_type[0] else False

    def build_task_head(self, cfg, task):
        return TaskHead(
            task,
            cfg.embed_dim,
            cfg.out_head,
            cfg.aggregation,
        )

    def forward(self, data, other_data=None):
        x = self.encoder(data)
        if hasattr(data, 'sub_index'):
            x = x[data.sub_index]
        other_x = other_data
        if self.encode_other_protein:
            assert other_data is not None, "other data should be provided"
            other_x = self.encoder(other_data)
            if hasattr(other_data, 'sub_index'):
                other_x = other_x[other_data.sub_index]

        output = self.task_head(x, other_x)
        return output

    def step(self, batch):
        if self.encode_other:
            if self.encode_other_protein:
                data, other_x, y = batch
                if hasattr(y, 'y'):
                    y = y.y
            else:
                data, other_x, y = batch, batch.other_x, batch.y
        else:
            data, other_x, y = batch, None, batch.y
        y_hat = self.forward(data, other_x)
        return y_hat, y

    def val_step(self, batch):
        if self.encode_other:
            if self.encode_other_protein:
                data, other_x, y = batch
                if hasattr(y, 'y'):
                    y = y.y
            else:
                data, other_x, y = batch, batch.other_x, batch.y
        else:
            data, other_x, y = batch, None, batch.y
        y_hat = self.forward(data, other_x)
        if y_hat.shape[0] > data.num_nodes:
            y_hat_new = []
            y_new = []
            total = 0
            for i in range(len(data.ptr) - 1):
                size_i = data.ptr[i + 1] - data.ptr[i]
                size_j = other_x.ptr[i + 1] - other_x.ptr[i]
                y_hat_new.append(y_hat[total: total + size_i * size_j].view(size_i, size_j))
                y_new.append(y[total: total + size_i * size_j].view(size_i, size_j))
                total += size_i * size_j
            y_hat = y_hat_new
            y = y_new
        return y_hat, y

    def from_pretrained(self, model_path):
        self.encoder.from_pretrained(model_path)

    def save(self, save_path):
        torch.save(
            {'cfg': self.cfg, 'state_dict': self.state_dict()},
            save_path
        )
