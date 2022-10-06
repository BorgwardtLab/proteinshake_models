import torch
import numpy as np
from src.models.voxel import VoxelNet_Pretrain
from src.util import Trainer, Evaluator, train_test_split
from proteinshake.datasets import AlphaFoldDataset, EnzymeCommissionDataset
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--epochs', type=int, default=1, help='Epochs')
parser.add_argument('--task', type=str, default='PT', help='Task')
parser.add_argument('--representation', type=str, default='voxel', help='Representation')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
args = parser.parse_args()

path = f'results/{args.representation}_{args.task}'
if args.pretrained:
    path += '_pretrained'

if args.task == 'PT': # pretraining
    ds = AlphaFoldDataset(root='data/af', organism='methanocaldococcus_jannaschii')
    model = VoxelNet_Pretrain()
    def transform(data, labels):
        nonzero = (data == 0).all(-1)
        volume = nonzero.sum()
        n, m = int(volume * 0.15), volume-int(volume * 0.15)
        mask = torch.zeros(data.shape[:-1]).bool()
        inner_mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(volume)].bool()
        mask[nonzero] = inner_mask
        masked = data.clone()
        masked[mask] = 1
        return data, masked, mask
    def criterion(batch, y_pred):
        data, masked, mask = batch
        y_pred = y_pred[mask]
        y_true = data[mask].cuda()
        return torch.nn.functional.mse_loss(y_pred, y_true)
    def metric(batch, y_pred):
        data, masked, mask = batch
        y_pred = torch.argmax(y_pred[mask], -1)
        y_true = torch.argmax(data[mask], -1).cuda()
        return torch.sum(y_pred == y_true) / y_true.shape[0]

if args.pretrained:
    model.from_pretrained(f'results/{args.representation}_PT/base_weights.pt')

if args.representation == 'voxel':
    ds = ds.to_voxel(gridsize=(50,50,50), voxelsize=10).torch(transform=transform)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
train, test = train_test_split(ds)
train = DataLoader(train, batch_size=32)
test = DataLoader(test, batch_size=32)
trainer = Trainer(model, optimizer, criterion, train, path)
evaluator = Evaluator(model, metric, test)
trainer.train(args.epochs)
evaluator.eval()
