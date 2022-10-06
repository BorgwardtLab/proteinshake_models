import torch
import torch.nn as nn

class VoxelNetBase(nn.Module):

    def __init__(self, input_dim=20, hidden_dim=128, num_layers=2, kernel_size=3, dropout=0.2):
        super().__init__()
        def block(n):
            i = input_dim if n == 0 else hidden_dim
            #o = output_dim if n == num_layers-1 else hidden_dim
            return nn.Sequential(
                nn.Conv3d(in_channels=i, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
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
        x = torch.permute(x, (0,4,1,2,3))
        for layer in self.layers:
            x = layer(x)
        return x

class VoxelNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = VoxelNetBase()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def from_pretrained(self, path):
        self.base.load_weights(path)

class VoxelNet_Pretrain(VoxelNet):

    def __init__(self, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv3d(in_channels=hidden_dim, out_channels=20, kernel_size=kernel_size, stride=1, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, batch):
        data, masked, mask = batch
        x = self.base(masked.cuda())
        return self.output(x).permute(0,2,3,4,1)


class VoxelNet_EC(VoxelNet):

    def __init__(self, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, batch):
        data, label = batch
        x = self.base(data.cuda())
        x = torch.amax(x.permute(0,2,3,4,1), dim=(1,2,3))
        return self.output(x)
