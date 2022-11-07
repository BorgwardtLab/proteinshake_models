import torch
import torch.nn as nn

hidden_dim = 256
kernel_size = 5

class VoxelNetBase(nn.Module):

    def __init__(self, input_dim=20, hidden_dim=hidden_dim, num_layers=3, kernel_size=kernel_size, dropout=0.2):
        super().__init__()
        def block(n):
            i = input_dim if n == 0 else hidden_dim
            #o = output_dim if n == num_layers-1 else hidden_dim
            return nn.Sequential(
                nn.Conv3d(in_channels=i, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
                nn.LeakyReLU(),
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

    def __init__(self, task, hidden_dim=hidden_dim, kernel_size=kernel_size):
        super().__init__()
        self.task = task
        self.output = nn.Sequential(
            nn.Conv3d(in_channels=hidden_dim, out_channels=20, kernel_size=kernel_size, stride=1, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, batch):
        data, masked, mask = batch
        x = self.base(masked.cuda())
        return self.output(x).permute(0,2,3,4,1)

    def transform(self, args):
        data, protein_dict = args
        nonzero = ~((data == 0).all(-1))
        volume = nonzero.sum()
        n, m = int(volume * 0.15), volume-int(volume * 0.15)
        mask = torch.zeros(data.shape[:-1]).bool()
        inner_mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(volume)].bool()
        mask[nonzero] = inner_mask
        masked = data.clone()
        masked[mask] = 1
        return data, masked, mask

    def criterion(self, batch, y_pred):
        data, masked, mask = batch
        y_pred = y_pred[mask]
        y_true = data[mask]
        #return torch.nn.functional.mse_loss(y_pred, y_true)
        return torch.nn.functional.cross_entropy(y_pred, torch.argmax(y_true,-1).cuda())

    def predict(self, batch):
        y_pred = self.forward(batch)
        data, masked, mask = batch
        y_pred = y_pred[mask]
        y_true = data[mask].cuda()
        #return torch.nn.functional.l1_loss(y_pred,y_true)
        y_pred = torch.argmax(y_pred,-1)
        y_true = torch.argmax(y_true,-1)
        return y_true, y_pred


class VoxelNet_EC(VoxelNet):

    def __init__(self, task, hidden_dim=hidden_dim, kernel_size=kernel_size, dropout=0.2):
        super().__init__()
        self.task = task
        self.output = nn.Sequential(
            nn.Linear(hidden_dim*10*10*10, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.task.num_classes),
            nn.LogSoftmax(-1)
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, batch):
        data, label = batch
        x = self.base(data.cuda())
        #print(x.shape)
        #x = self.pool(x).view(x.shape[0],x.shape[1])#torch.amax(x.permute(0,2,3,4,1), dim=(1,2,3))
        #return self.output(x)
        x = self.pool(x)
        return self.output(x.reshape(x.shape[0],-1))

    def transform(self, args):
        data, protein_dict = args
        return data, torch.eye(self.task.num_classes)[self.task.target(protein_dict)].float()

    def criterion(self, batch, y_pred):
        data, label = batch
        return torch.nn.functional.nll_loss(y_pred, torch.argmax(label,-1).cuda())

    def predict(self, batch):
        y_pred = self.forward(batch)
        data, label = batch
        y_pred = torch.argmax(y_pred,-1).cpu()
        y_true = torch.argmax(label,-1)
        return y_true, y_pred

class VoxelNet_LA(VoxelNet):

    def __init__(self, task, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.task = task
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=i, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv3d(in_channels=i, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        data, label = batch
        x = self.base(data.cuda())
        x = torch.amax(self.layer(x).permute(0,2,3,4,1), dim=(1,2,3))
        return self.output(x)

    def transform(self, args):
        data, protein_dict = args
        return data, torch.tensor(self.task.target(protein_dict)).float()

    def criterion(self, batch, y_pred):
        data, label = batch
        return torch.nn.functional.mse_loss(y_pred.squeeze(-1), label.cuda())

    def predict(self, batch):
        y_pred = self.forward(batch)
        data, y_true = batch
        return y_true, y_pred.squeeze(-1)
