import torch
import torch.nn as nn
import torch.nn.functional as F

# modified from: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch/notebook

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


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
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64-20,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)


        self.bn1 = nn.BatchNorm1d(64-20)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

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
            nn.Linear(1024,20),
            nn.Sigmoid()
        )

    def forward(self, batch):
        coords, labels, masked, mask = batch
        coords = coords.permute(0,2,1)
        labels = labels.permute(0,2,1)
        x, matrix3x3, matrix64x64 = self.base(coords.cuda(), labels.cuda())
        return self.output(x)


class PointNet_EC(PointNet):

    def __init__(self, hidden_dim=128, kernel_size=3):
        super().__init__()
        classes = 7
        self.output = nn.Sequential(
            nn.Linear(1024, 7),
        )

    def forward(self, batch):
        coords, labels, ec = batch
        coords = coords.permute(0,2,1)
        labels = labels.permute(0,2,1)
        x, matrix3x3, matrix64x64 = self.base(coords.cuda(), labels.cuda())
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        return self.output(x)
