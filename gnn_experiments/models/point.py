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

    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=d1)
        self.conv1 = nn.Conv1d(3,d1-20,1)

        self.conv2 = nn.Conv1d(64,d2,1)
        self.conv3 = nn.Conv1d(d2,d3,1)


        self.bn1 = nn.BatchNorm1d(d1-20)
        self.bn2 = nn.BatchNorm1d(d2)
        self.bn3 = nn.BatchNorm1d(d3)


    def forward(self, input, labels):
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        xb = F.relu(self.bn1(self.conv1(xb)))
        xb = torch.cat([xb,labels], dim=1)
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        return xb, matrix3x3, matrix64x64


class PointNet_Pretraining(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = PointNetBase()
        self.head = nn.Sequential(
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
        return self.head(x).permute(0,2,1)


class PointNet_EnzymeClass(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = PointNetBase()
        self.head = nn.Sequential(
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


class PointNet_LigandAffinity(PointNet):

    def __init__(self, task, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.task = task
        self.head = nn.Sequential(
            nn.Linear(d3, 1),
        )

    def forward(self, batch):
        coords, labels, ec = batch
        coords = coords.permute(0,2,1)
        labels = labels.permute(0,2,1)
        x, matrix3x3, matrix64x64 = self.base(coords.cuda(), labels.cuda())
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        return self.output(x)
