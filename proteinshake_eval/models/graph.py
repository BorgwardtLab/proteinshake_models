import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils
from ..position_encoder import build_position_encoding


class Aggregator(nn.Module):
    def __init__(self, embed_dim=256, aggregation='concat', normalize=False):
        super().__init__()
        self.aggregation = aggregation
        self.normalize = normalize

        if aggregation == 'concat':
            self.aggregator = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(True),
                nn.Linear(embed_dim, embed_dim)
            )
        elif aggregation == 'dot' or aggregation == 'sum':
            self.aggregator = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(True),
                nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, x1, x2):
        if self.normalize:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        if self.aggregation == 'concat':
            x = torch.cat((x1, x2), dim=-1)
        elif self.aggregation == 'dot':
            x = x1 * x2
        elif self.aggregation == 'sum':
            x = x1 + x2
        return self.aggregator(x)


class GINConv(gnn.MessagePassing):
    def __init__(self, embed_dim=256, use_edge_attr=False):
        super().__init__(aggr='add')

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.BatchNorm1d(2 * embed_dim),
            nn.ReLU(True),
            nn.Linear(2 * embed_dim, embed_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            self.edge_encoder = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        out = self.mlp(
            (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class GCNConv(gnn.MessagePassing):
    def __init__(self, embed_dim=256, use_edge_attr=False):
        super().__init__(aggr='add')
        self.use_edge_attr = use_edge_attr

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)
        self.edge_encoder = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.linear(x)
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        row, col = edge_index

        deg = utils.degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr = edge_attr, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class SAGEConv(gnn.MessagePassing):
    def __init__(self, embed_dim, aggr='add', use_edge_attr=False):
        super(SAGEConv, self).__init__(aggr=aggr)
        self.use_edge_attr = use_edge_attr

        self.lin_l = nn.Linear(embed_dim, embed_dim)
        self.lin_r = nn.Linear(embed_dim, embed_dim)

        self.edge_encoder = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_l(out)

        x_r = x[1]
        out += self.lin_r(x_r)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


GET_GNN_ENCODER = {
    'gin': GINConv,
    'gcn': GCNConv,
    'sage': SAGEConv,
}
GNN_TYPES = GET_GNN_ENCODER.keys()
NUM_PROTEINS = 20
NUM_PROTEINS_MASK = NUM_PROTEINS + 1


class GNN(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3, dropout=0.0, gnn_type='gin',
                 use_edge_attr=False, pe=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # self.x_embedding = nn.Linear(NUM_PROTEINS_MASK, embed_dim, bias=False)
        self.x_embedding = nn.Embedding(NUM_PROTEINS_MASK, embed_dim)

        self.position_embedding = build_position_encoding(embed_dim, pe)

        gnn_model = GET_GNN_ENCODER[gnn_type]
        self.gnns = nn.ModuleList()
        for _ in range(num_layers):
            self.gnns.append(gnn_model(embed_dim, use_edge_attr=use_edge_attr))

        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(embed_dim))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        output = self.x_embedding(x)
        if self.position_embedding is not None:
            pos = self.position_embedding(data)
            output = output + pos

        for layer in range(self.num_layers):
            output = self.gnns[layer](output, edge_index, edge_attr)
            output = self.batch_norms[layer](output)

            if layer == self.num_layers - 1:
                output = F.dropout(output, self.dropout, training=self.training)
            else:
                output = F.dropout(F.relu(output), self.dropout, training=self.training)

        return output

    def save(self, model_path, args):
        torch.save(
            {'args': args, 'state_dict': self.state_dict()},
            model_path
        )


class NodeClassifier(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3, dropout=0.0, gnn_type='gin',
                 use_edge_attr=False, pe=None, num_class=20):
        super().__init__()
        self.base = GNN(embed_dim, num_layers, dropout, gnn_type, use_edge_attr, pe)
        self.head = nn.Linear(embed_dim, num_class)

    def step(self, batch):
        node_true = batch.masked_node_label
        node_repr = self.base(batch)
        node_pred = self.head(node_repr[batch.masked_node_indices])
        return node_pred, node_true

    def save(self, path, args):
        torch.save(
            {'args': args, 'state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
            path
        )


class GNN_nodepred(nn.Module):
    def __init__(self, num_class, embed_dim=256, num_layers=3, dropout=0.0, gnn_type='gin',
                 use_edge_attr=False, pe=None, out_head='mlp'):
        super().__init__()

        self.encoder = GNN(embed_dim, num_layers, dropout, gnn_type, use_edge_attr, pe)

        if out_head == 'linear':
            self.out_head = nn.Linear(embed_dim, num_class)
        else:
            self.out_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(True),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.ReLU(True),
                nn.Linear(embed_dim // 4, num_class)
            )

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path)['state_dict'])

    def forward(self, data):
        bsz = len(data.ptr) - 1
        output = self.encoder(data)
        return self.classifier(output)


class GNN_encoder(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3, dropout=0.0, gnn_type='gin',
                 use_edge_attr=False, pe=None, global_pool=None):
        super().__init__()

        self.encoder = GNN(embed_dim, num_layers, dropout, gnn_type, use_edge_attr, pe)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'max':
            self.pooling = gnn.global_max_pool
        elif global_pool is None:
            self.pooling = None

    def forward(self, data):
        bsz = len(data.ptr) - 1
        output = self.encoder(data)
        if self.pooling is not None:
            output = self.pooling(output, data.batch)
        return output

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path)['state_dict'])


class GNN_graphpred(nn.Module):
    def __init__(self, num_class, embed_dim=256, num_layers=3, dropout=0.0, gnn_type='gin',
                 use_edge_attr=False, pe=None, global_pool='mean', out_head='mlp',
                 pair_prediction=False, same_type=False, other_dim=1024,
                 aggregation='dot', aggregation_norm=False):
        super().__init__()

        self.encoder = GNN(embed_dim, num_layers, dropout, gnn_type, use_edge_attr, pe)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'max':
            self.pooling = gnn.global_max_pool
        elif global_pool is None:
            self.pooling = None

        self.pair_prediction = pair_prediction
        self.same_type = same_type
        self.aggregation = aggregation
        if pair_prediction:
            if not same_type:
                self.other_encoder = nn.Sequential(
                    nn.Linear(other_dim, embed_dim),
                    nn.BatchNorm1d(embed_dim),
                    nn.ReLU(True),
                    nn.Linear(embed_dim, embed_dim),
                    nn.BatchNorm1d(embed_dim),
                )
            self.aggregator = Aggregator(embed_dim, aggregation, aggregation_norm)


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

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path)['state_dict'])
        print(f"Model loaded from {model_path}")

    def forward(self, data, other_data = None):
        bsz = len(data.ptr) - 1
        output = self.encoder(data)
        if self.pooling is not None:
            output = self.pooling(output, data.batch)

        if self.pair_prediction:
            assert other_data is not None, "other_data should be provided!"
            if self.same_type:
                other_output = self.encoder(other_data)
                other_output = self.pooling(other_output, other_data.batch)
            else:
                other_output = self.other_encoder(other_data)
            output = self.aggregator(output, other_output)

        return self.classifier(output)

    def save(self, model_path, args):
        torch.save(
            {'args': args, 'state_dict': self.state_dict()},
            model_path
        )

    def step(self, batch):
        if self.pair_prediction:
            if self.same_type:
                data, other_x, y = batch
            else:
                data, other_x, y = batch, batch.other_x, batch.y
        else:
            data, other_x, y = batch, None, batch.y
        y_hat = self.forward(data, other_x)
        return y_hat, y
