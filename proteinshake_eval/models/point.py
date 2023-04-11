import torch
import torch.nn as nn
import torch.nn.functional as F
from .aggregator import Aggregator, build_pooling

# modified from: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch/notebook

NUM_PROTEINS = 20
NUM_PROTEINS_MASK = NUM_PROTEINS + 1


class Tnet(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(input_dim, embed_dim, 1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim * 2, 1)
        self.conv3 = nn.Conv1d(embed_dim * 2, embed_dim * 8, 1)
        self.fc1 = nn.Linear(embed_dim * 8, embed_dim * 2)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim, input_dim * input_dim)

        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim * 2)
        self.bn3 = nn.BatchNorm1d(embed_dim * 8)
        self.bn4 = nn.BatchNorm1d(embed_dim * 2)
        self.bn5 = nn.BatchNorm1d(embed_dim)
        self.d3 = embed_dim * 8

        self.register_buffer("init", torch.eye(input_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # x.shape == (bs, 3, n)
        bs = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.d3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        # init = torch.eye(self.input_dim, requires_grad=True).repeat(bs, 1, 1).to(x.device)
        init = self.init.repeat(bs, 1, 1)
        matrix = self.fc3(x).view(-1, self.input_dim, self.input_dim) + init
        return matrix

class PointNetBase(nn.Module):

    def __init__(self, input_dim=3, embed_dim=64):
        super().__init__()
        self.input_transform = Tnet(input_dim=input_dim)
        self.embedding = nn.Embedding(
            NUM_PROTEINS_MASK + 1, embed_dim, padding_idx= NUM_PROTEINS_MASK)
        self.conv_merge = nn.Conv1d(embed_dim, embed_dim, 1)
        self.feature_transform = Tnet(input_dim=embed_dim)
        self.conv1 = nn.Conv1d(3, embed_dim, 1)
        self.conv2 = nn.Conv1d(embed_dim, 2 * embed_dim, 1)
        self.conv3 = nn.Conv1d(2 * embed_dim, 8 * embed_dim, 1)
        self.out_head = nn.Sequential(
            nn.Conv1d(8 * embed_dim, 2 * embed_dim, 1),
            nn.BatchNorm1d(2 * embed_dim),
            nn.ReLU(True),
            nn.Conv1d(2 * embed_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
        )

        self.bn_merge = nn.BatchNorm1d(embed_dim)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim * 2)
        self.bn3 = nn.BatchNorm1d(embed_dim * 8)

        self.register_buffer("ident1", torch.eye(input_dim).view(1, input_dim, input_dim))
        self.register_buffer("ident2", torch.eye(embed_dim).view(1, embed_dim, embed_dim))
        self.regularizer_loss_ = None

    def forward(self, x, labels):
        matrix3x3 = self.input_transform(x)
        x = x.transpose(1, 2)
        x = torch.bmm(x, matrix3x3).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))

        labels = self.embedding(labels).transpose(1, 2)
        x = F.relu(self.bn_merge(self.conv_merge(x + labels)))
        matrix64x64 = self.feature_transform(x)
        x = x.transpose(1, 2)
        x = torch.bmm(x, matrix64x64).transpose(1, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.out_head(F.relu(x)).transpose(1, 2)
        self.regularizer_loss_ = self.regularizer(matrix3x3, matrix64x64)
        return x

    def regularizer(self, matrix3x3, matrix64x64):
        loss1 = F.mse_loss(torch.bmm(
            matrix3x3, matrix3x3.transpose(1, 2)), self.ident1.expand_as(matrix3x3))
        loss2 = F.mse_loss(torch.bmm(
            matrix64x64, matrix64x64.transpose(1, 2)), self.ident2.expand_as(matrix64x64))
        return loss1 + loss2

    def regularizer_loss(self, alpha=0.001):
        return alpha * self.regularizer_loss_


class PointNet_encoder(nn.Module):
    def __init__(self, embed_dim=64, global_pool='max', alpha=0.0001):
        super().__init__()
        self.encoder = PointNetBase(embed_dim=embed_dim)
        self.alpha = alpha

        self.pooling = build_pooling(global_pool, dim=1)

    def forward(self, data, other_data=None):
        x, labels, mask = data.coords, data.labels, data.mask
        output = self.encoder(x, labels)
        if self.pooling is not None:
            output = self.pooling(output, mask)
        else:
            if not hasattr(data, "masked_indices"):
                output = output[mask]

        return output

    def regularizer_loss(self):
        return self.encoder.regularizer_loss(self.alpha)

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path)['state_dict'])
