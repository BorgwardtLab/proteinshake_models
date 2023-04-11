import torch
import torch.nn as nn
from .aggregator import build_pooling


NUM_PROTEINS = 20
NUM_PROTEINS_MASK = NUM_PROTEINS + 1


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
        output = self.encoder(data.x).permute(0, 2, 3, 4, 1)
        if self.pooling is not None:
            output = self.pooling(output, data.mask)
        return output

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path)['state_dict'])
