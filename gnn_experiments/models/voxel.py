import torch
import torch.nn as nn


class VoxelNetBase(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout):
        super().__init__()
        def block(n):
            in_channels = input_dim if n == 0 else hidden_dim
            return nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
        self.layers = nn.ModuleList([block(n) for n in range(num_layers)])

    def forward(self, x):
        x = torch.permute(x, (0,4,1,2,3))
        for layer in self.layers:
            x = layer(x)
        return x

class VoxelNet_Pretraining(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.head = nn.Linear(hidden_dim, 20)

    def step(self, batch):
        data, masked, mask = batch
        node_embs = self.base(masked).permute(0,2,3,4,1)
        node_embs = node_embs[mask]
        y_hat = self.head(node_embs)
        y = data[mask]
        return y_hat, torch.argmax(y, -1) # each voxel can have multiple amino acids, so the last dimension is a float value indicating the fraction of amino acid X in the voxel

    def save(self, path, args):
        torch.save(
            {'args': args, 'base_state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
            path
        )


class VoxelNet_EnzymeClass(nn.Module):

    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout, other_dim):
        super().__init__()
        self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.pool = nn.MaxPool3d(20)
        self.head = nn.Linear(hidden_dim, out_dim)

    def step(self, batch):
        data, y = batch
        nonzero_mask = ~((data == 0).all(-1))
        x = self.base(data).permute(0,2,3,4,1)
        x[nonzero_mask] = 0.
        x = torch.amax(x, dim=(1,2,3))
        y_hat = self.head(x)
        return y_hat, torch.argmax(y, -1)

    def save(self, path, args):
        torch.save(
            {'args': args, 'base_state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
            path
        )


class VoxelNet_LigandAffinity(nn.Module):

    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout, other_dim):
        super().__init__()
        self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.pool = nn.MaxPool3d(20)
        self.head = nn.Linear(hidden_dim+other_dim, out_dim)

    def step(self, batch):
        data, y, fingerprint = batch
        nonzero_mask = ~((data == 0).all(-1))
        x = self.base(data).permute(0,2,3,4,1)
        x[nonzero_mask] = 0.
        x = torch.amax(x, dim=(1,2,3))
        x = torch.cat([x,fingerprint], dim=-1)
        y_hat = self.head(x)
        return y_hat, torch.unsqueeze(y,1)

    def save(self, path, args):
        torch.save(
            {'args': args, 'base_state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
            path
        )
