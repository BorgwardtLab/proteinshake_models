import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils


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
                 use_edge_attr=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # self.x_embedding = nn.Linear(NUM_PROTEINS_MASK, embed_dim, bias=False)
        self.x_embedding = nn.Embedding(NUM_PROTEINS_MASK, embed_dim)

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
    def __init__(self, embed_dim, num_class=NUM_PROTEINS):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        return self.classifier(x)


class GNN_nodepred(nn.Module):
    def __init__(self, num_class, embed_dim=256, num_layers=3, dropout=0.0, gnn_type='gin',
                 use_edge_attr=False, out_head='mlp'):
        super().__init__()

        self.encoder = GNN(embed_dim, num_layers, dropout, gnn_type, use_edge_attr)

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
        self.gnn.load_state_dict(torch.load(model_path))

    def forward(self, data):
        bsz = len(data.ptr) - 1
        output = self.encoder(data)
        return self.classifier(output)


class GNN_graphpred(nn.Module):
    def __init__(self, num_class, embed_dim=256, num_layers=3, dropout=0.0, gnn_type='gin',
                 use_edge_attr=False, global_pool='mean', out_head='mlp'):
        super().__init__()

        self.encoder = GNN(embed_dim, num_layers, dropout, gnn_type, use_edge_attr)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.pooling = None

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
        self.gnn.load_state_dict(torch.load(model_path))

    def forward(self, data):
        bsz = len(data.ptr) - 1
        output = self.encoder(data)

        output = self.pooling(output, data.batch)
        return self.classifier(output)
