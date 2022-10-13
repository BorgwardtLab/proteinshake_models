import torch
import torch.nn as nn
import torch.nn.functional as F

# modified from: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch/notebook

d1 = 64
d2 = 512
d3 = 2048

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k,d1,1)
        self.conv2 = nn.Conv1d(d1,d2,1)
        self.conv3 = nn.Conv1d(d2,d3,1)
        self.fc1 = nn.Linear(d3,d2)
        self.fc2 = nn.Linear(d2,d1)
        self.fc3 = nn.Linear(d1,k*k)

        self.bn1 = nn.BatchNorm1d(d1)
        self.bn2 = nn.BatchNorm1d(d2)
        self.bn3 = nn.BatchNorm1d(d3)
        self.bn4 = nn.BatchNorm1d(d2)
        self.bn5 = nn.BatchNorm1d(d1)


    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1).cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class PointNetBase(nn.Module):

    def __init__(self, input_dim=20, hidden_dim=128, num_layers=2, kernel_size=3, dropout=0.2):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=d1)
        self.conv1 = nn.Conv1d(3,d1-20,1)

        self.conv2 = nn.Conv1d(64,d2,1)
        self.conv3 = nn.Conv1d(d2,d3,1)


        self.bn1 = nn.BatchNorm1d(d1-20)
        self.bn2 = nn.BatchNorm1d(d2)
        self.bn3 = nn.BatchNorm1d(d3)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input, labels):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        xb = torch.cat([xb,labels], dim=1)

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        #xb = nn.MaxPool1d(xb.size(-1))(xb)
        #output = nn.Flatten(1)(xb)
        #return output, matrix3x3, matrix64x64
        return xb, matrix3x3, matrix64x64



class PointNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = PointNetBase()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def from_pretrained(self, path):
        self.base.load_weights(path)

class PointNet_Pretrain(PointNet):

    def __init__(self, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv1d(in_channels=d3*2, out_channels=20, kernel_size=1, stride=1, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, batch):
        coords, labels, masked, mask = batch
        coords = coords.permute(0,2,1)
        masked = masked.permute(0,2,1)
        x, matrix3x3, matrix64x64 = self.base(coords.cuda(), masked.cuda())
        g = nn.MaxPool1d(x.size(-1))(x)
        g = nn.Flatten(1)(g)
        x = torch.cat([x,g.unsqueeze(-1).repeat(1,1,x.shape[-1])], dim=1)
        return self.output(x).permute(0,2,1)

    def transform(self, coords, labels, protein_dict):
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        length = labels.shape[0]
        n, m = int(length * 0.15), length-int(length * 0.15)
        mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(length)].bool()
        masked = labels.clone()
        masked[mask] = torch.ones(labels.shape[1]).float()
        return coords, labels, masked, mask

    def criterion(self, batch, y_pred):
        coords, labels, masked, mask = batch
        y_pred = y_pred[mask]
        y_true = torch.argmax(labels[mask], -1).cuda()
        return torch.nn.functional.cross_entropy(y_pred, y_true)

    def metric(self, batch, y_pred):
        coords, labels, masked, mask = batch
        y_pred = torch.argmax(y_pred[mask], -1)
        y_true = torch.argmax(labels[mask], -1).cuda()
        return torch.sum(y_pred == y_true) / y_true.shape[0]


class PointNet_EC(PointNet):

    def __init__(self, hidden_dim=128, kernel_size=3):
        super().__init__()
        classes = 7
        self.output = nn.Sequential(
            nn.Linear(d3, 7),
        )

    def forward(self, batch):
        coords, labels, ec = batch
        coords = coords.permute(0,2,1)
        labels = labels.permute(0,2,1)
        x, matrix3x3, matrix64x64 = self.base(coords.cuda(), labels.cuda())
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        return self.output(x)

    def transform(self, coords, labels, protein_dict):
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        return coords, labels, torch.eye(7)[int(protein_dict['protein']['EC'].split('.')[0])-1].float()

    def criterion(self, batch, y_pred):
        coords, labels, ec = batch
        return torch.nn.functional.cross_entropy(y_pred, torch.argmax(ec,-1).cuda())

    def metric(self, batch, y_pred):
        coords, labels, ec = batch
        y_pred = torch.argmax(y_pred,-1)
        ec = torch.argmax(ec,-1).cuda()
        return torch.sum(y_pred == ec) / ec.shape[0]
