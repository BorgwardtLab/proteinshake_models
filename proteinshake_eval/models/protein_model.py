import torch
from torch import nn
from .aggregator import Aggregator
from .graph import GNN_encoder
from .point import PointNet_encoder
from .voxel import VoxelNet_encoder


class TaskHead(nn.Module):
    def __init__(self, task, embed_dim=256, out_head='linear',
                 aggregation='dot'):
        super().__init__()

        task_level, task_type = task.task_type

        self.pair_prediction = task.pair_data

        if self.pair_prediction:
            if task_level != 'protein_pair':
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


class ProteinStructureNet(nn.Module):
    def __init__(self, cfg, task):
        super().__init__()
        self.cfg = cfg
        # Build protein encoder
        self.encoder = self.build_encoder(cfg)

        # Build task head
        self.task_head = self.build_task_head(cfg, task)

        self.encode_other = task.pair_data
        self.encode_other_protein = task.task_type[0] == 'protein_pair'

    def build_encoder(self, cfg):
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
            return PointNet_encoder(
                cfg.embed_dim,
                cfg.pooling,
                cfg.alpha
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

    def build_task_head(self, cfg, task):
        return TaskHead(
            task,
            cfg.embed_dim,
            cfg.out_head,
            cfg.aggregation,
        )

    def forward(self, data, other_data=None):
        x = self.encoder(data)
        other_x = other_data
        if self.encode_other_protein:
            assert other_data is not None, "other data should be provided"
            other_x = self.encoder(other_data)

        output = self.task_head(x, other_x)
        return output

    def step(self, batch):
        if self.encode_other:
            if self.encode_other_protein:
                data, other_x, y = batch
            else:
                data, other_x, y = batch, batch.other_x, batch.y
        else:
            data, other_x, y = batch, None, batch.y
        y_hat = self.forward(data, other_x)
        return y_hat, y

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path)['state_dict'])
        print(f"Model loaded from {model_path}")

    def save(self, save_path):
        torch.save(
            {'cfg': self.cfg, 'state_dict': self.state_dict()},
            save_path
        )
