import torch
import torch.nn as nn

class VoxelNetBase(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout):
        def block(n):
            i = input_dim if n == 0 else hidden_dim
            #o = output_dim if n == num_layers-1 else hidden_dim
            return nn.Sequential(
                nn.Conv3d(in_channels=i, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
                nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        self.layers = nn.ModuleList([block(n) for n in range(num_layers)])

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        x = self.input(x)
        for layer in self.layers:
            x = layer(x)
        return x


class VoxelNetHead(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(torch.flatten(x,1))


class VoxelNet(nn.Module):

    def __init__(self, level='residue', input_dim, output_dim, volume, hidden_dim=128, num_layers=2, kernel_size=3, dropout=0.2):
        self.base = VoxelNetBase(input_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.head = VoxelNetHead(hidden_dim*volume, output_dim)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x
