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

    def forward(self, x):
        x = self.base(x)
        return self.output(x).permute(0,2,3,4,1)

    def step(self, batch):
        data, masked, mask = batch
        node_embs = self.base(masked)
        node_embs = node_embs[mask].view(-1, self.hidden_dim)
        y_hat = self.head(node_embs)
        y = data[mask].view(-1, 20)
        return y_hat, y

    def save(self, path, args):
        torch.save(
            {'args': args, 'base_state_dict': self.base.state_dict(), 'head_state_dict': self.head.state_dict()},
            path
        )


class VoxelNet_EnzymeClass(nn.Module):

    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.pool = nn.MaxPool3d(20)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*10*10*10, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LogSoftmax(-1)
        )

    def forward(self, x):
        return self.head(self.pool(x))


class VoxelNet_LigandAffinity(nn.Module):

    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.pool = nn.MaxPool3d(20)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.head(self.pool(x))
