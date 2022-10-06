import torch
import numpy as np
from src.models.voxel import VoxelNet_Pretrain, VoxelNet_EC
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
        volume = torch.prod(torch.tensor(data.shape[:-1]))
        n, m = int(volume * 0.5), volume-int(volume * 0.5)
        mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(volume)].reshape(*data.shape[:-1]).bool()
        masked = data.clone()
        masked[mask] = 1
        targets = data[mask]
        return masked, targets, mask
    def criterion(batch, y_pred):
        masked, targets, mask = batch
        y_pred = y_pred[mask]
        y_true = torch.flatten(targets, end_dim=-2).cuda()
        y_pred = y_pred[y_true != torch.zeros(y_true.shape[-1]).cuda()]
        y_true = y_true[y_true != torch.zeros(y_true.shape[-1]).cuda()]
        return torch.nn.functional.mse_loss(y_pred, y_true)
    def metric(batch, y_pred):
        masked, targets, mask = batch
        y_pred = torch.argmax(y_pred[mask], -1)
        y_true = torch.argmax(torch.flatten(targets, end_dim=-2), -1).cuda()
        y_pred = y_pred[y_true != torch.zeros(y_true.shape[-1]).cuda()]
        y_true = y_true[y_true != torch.zeros(y_true.shape[-1]).cuda()]
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
