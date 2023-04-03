import torch
import torch.nn as nn
from .aggregator import build_pooling


NUM_PROTEINS = 20
NUM_PROTEINS_MASK = NUM_PROTEINS + 1

def mask_empty_voxels(x, data):
    nonzero_mask = ~((data == 0).all(-1))
    x = x.permute(0,2,3,4,1)
    x[nonzero_mask] = 0.
    x = torch.amax(x, dim=(1,2,3))
    return x

# class Template(nn.Module):

#     def save(self, path, args):
#         torch.save({'args': args, 'state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()}, path)

#     def from_pretrained(self, path):
#         self.base.load_state_dict(torch.load(path)['state_dict'])


class VoxelNetBase(nn.Module):

    def __init__(self, embed_dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.layers = nn.ModuleList([self.block(n) for n in range(num_layers)])

    def block(self, n):
        in_channels = NUM_PROTEINS if n == 0 else self.embed_dim
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=self.embed_dim,
                kernel_size=self.kernel_size, padding='same'
            ),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = torch.permute(x, (0,4,1,2,3))
        for layer in self.layers:
            x = layer(x)
        return x


class VoxelNet_encoder(nn.Module):
    def __init__(self, embed_dim, num_layers, kernel_size, dropout=0.0, pooling='max'):
        super().__init__()
        self.encoder = VoxelNetBase(
            embed_dim, num_layers, kernel_size, dropout
        )
        self.pooling = build_pooling(pooling, dim=(1, 2, 3))

    def forward(self, data):
        x, mask = data.x, data.mask
        output = self.encoder(x).permute(0, 2, 3, 4, 1)
        if self.pooling is not None:
            output = self.pooling(output, mask)
        return output

# class VoxelNet_Pretraining(Template):

#     def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout):
#         super().__init__()
#         self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
#         self.head = nn.Linear(hidden_dim, 20)

#     def step(self, batch):
#         data, masked, mask = batch
#         node_embs = self.base(masked).permute(0,2,3,4,1)
#         node_embs = node_embs[mask]
#         y_hat = self.head(node_embs)
#         y = data[mask]
#         return y_hat, torch.argmax(y, -1) # each voxel can have multiple amino acids, so the last dimension is a float value indicating the fraction of amino acid X in the voxel




# class VoxelNet_EnzymeClass(Template):

#     def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout, other_dim):
#         super().__init__()
#         self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
#         self.head = nn.Linear(hidden_dim, out_dim)

#     def step(self, batch):
#         data, y = batch
#         nonzero_mask = ~((data == 0).all(-1))
#         x = self.base(data).permute(0,2,3,4,1)
#         x[nonzero_mask] = 0.
#         x = torch.amax(x, dim=(1,2,3))
#         y_hat = self.head(x)
#         return y_hat, torch.argmax(y, -1)

#     def save(self, path, args):
#         torch.save(
#             {'args': args, 'state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
#             path
#         )


# class VoxelNet_LigandAffinity(Template):

#     def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout, other_dim):
#         super().__init__()
#         self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
#         self.head = nn.Linear(hidden_dim+other_dim, out_dim)

#     def step(self, batch):
#         data, y, fingerprint = batch
#         nonzero_mask = ~((data == 0).all(-1))
#         x = self.base(data).permute(0,2,3,4,1)
#         x[nonzero_mask] = 0.
#         x = torch.amax(x, dim=(1,2,3))
#         x = torch.cat([x,fingerprint], dim=-1)
#         y_hat = self.head(x)
#         return y_hat, torch.unsqueeze(y,1)

#     def save(self, path, args):
#         torch.save(
#             {'args': args, 'state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
#             path
#         )


# class VoxelNet_BindingSite(Template):

#     def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout, other_dim):
#         super().__init__()
#         self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
#         self.head = nn.Linear(hidden_dim, out_dim)

#     def step(self, batch):
#         data, y = batch
#         nonzero_mask = ~((data == 0).all(-1))
#         x = self.base(data).permute(0,2,3,4,1)
#         x = x[nonzero_mask]
#         #... unfinished
#         y = y[nonzero_mask]
#         y_hat = self.head(x)
#         return y_hat, torch.argmax(y, -1)

#     def save(self, path, args):
#         torch.save(
#             {'args': args, 'state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
#             path
#         )

# class VoxelNet_Scop(Template):

#     def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout, other_dim):
#         super().__init__()
#         self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
#         self.head = nn.Linear(hidden_dim, out_dim)

#     def step(self, batch):
#         data, y = batch
#         x = mask_empty_voxels(self.base(data), data)
#         y_hat = self.head(x)
#         return y_hat, torch.argmax(y, -1)

#     def save(self, path, args):
#         torch.save(
#             {'args': args, 'state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
#             path
#         )

# class VoxelNet_Tm(Template):

#     def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout):
#         super().__init__()
#         self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
#         self.head = nn.Linear(hidden_dim*2, out_dim)

#     def step(self, batch):
#         data1, data2, y = batch
#         x1 = mask_empty_voxels(self.base(data1), data1)
#         x2 = mask_empty_voxels(self.base(data2), data2)
#         x = torch.cat([x1,x2], dim=-1)
#         y_hat = self.head(x)
#         return y_hat, y

#     def save(self, path, args):
#         torch.save(
#             {'args': args, 'state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
#             path
#         )
