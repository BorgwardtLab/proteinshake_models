import torch
import numpy as np
from src.models.voxel import *
from src.models.point import *
from src.util import *
from proteinshake.datasets import AlphaFoldDataset, EnzymeCommissionDataset
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--epochs', type=int, default=1, help='Epochs')
parser.add_argument('--task', type=str, default='PT', help='Task')
parser.add_argument('--representation', type=str, default='point', help='Representation')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
parser.add_argument('--batchsize', type=int, default=32, help='Batchsize')
args = parser.parse_args()

path = f'results/{args.representation}_{args.task}'
if args.pretrained:
    path += '_pretrained'

if args.task == 'PT': # pretraining
    ds = AlphaFoldDataset(root='data/af_homo_sapiens', organism='homo_sapiens')
    if args.representation == 'voxel':
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
            #y_pred = torch.argmax(y_pred[mask], -1)
            #y_true = torch.argmax(data[mask], -1).cuda()
            y_pred = y_pred[mask]
            y_true = data[mask].cuda()
            return torch.nn.functional.l1_loss(y_pred,y_true)#torch.sum(y_pred == y_true) / y_true.shape[0]
    elif args.representation == 'point':
        model = PointNet_Pretrain()
        L = 1024
        def transform(coords, labels, protein_dict):
            coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
            labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
            length = labels.shape[0]
            n, m = int(length * 0.15), length-int(length * 0.15)
            mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(length)].bool()
            masked = labels.clone()
            masked[mask] = torch.ones(labels.shape[1]).float()
            return coords, labels, masked, mask
        def criterion(batch, y_pred):
            coords, labels, masked, mask = batch
            y_pred = y_pred[mask]
            y_true = torch.argmax(labels[mask], -1).cuda()
            return torch.nn.functional.cross_entropy(y_pred, y_true)
        def metric(batch, y_pred):
            coords, labels, masked, mask = batch
            y_pred = y_pred[mask]
            y_true = labels[mask].cuda()
            return torch.sum(y_pred == y_true) / y_true.shape[0]

elif args.task == 'EC':
    ds = EnzymeCommissionDataset(root='data/ec')
    if args.representation == 'voxel':
        model = VoxelNet_EC()
        def transform(data, labels):
            return data, torch.eye(7)[int(labels['protein']['EC'].split('.')[0])-1].float()
        def criterion(batch, y_pred):
            data, label = batch
            return torch.nn.functional.cross_entropy(y_pred, torch.argmax(label,-1).cuda())
        def metric(batch, y_pred):
            data, label = batch
            y_pred = torch.argmax(y_pred,-1)
            label = torch.argmax(label,-1).cuda()
            return torch.sum(y_pred == label) / label.shape[0]
    elif args.representation == 'point':
        model = PointNet_EC()
        L = 1024
        def transform(coords, labels, protein_dict):
            coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
            labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
            return coords, labels, torch.eye(7)[int(protein_dict['protein']['EC'].split('.')[0])-1].float()
        def criterion(batch, y_pred):
            coords, labels, ec = batch
            return torch.nn.functional.cross_entropy(y_pred, torch.argmax(ec,-1).cuda())
        def metric(batch, y_pred):
            coords, labels, ec = batch
            y_pred = torch.argmax(y_pred,-1)
            ec = torch.argmax(ec,-1).cuda()
            return torch.sum(y_pred == ec) / ec.shape[0]

if args.pretrained:
    print('Loading pretrained weights')
    model.from_pretrained(f'results/{args.representation}_PT/base_weights.pt')

if args.representation == 'voxel':
    ds = ds.to_voxel(gridsize=(50,50,50), voxelsize=10).torch(transform=transform)
if args.representation == 'point':
    ds = ds.to_point().torch(transform=transform)

#ds = torch.utils.data.Subset(ds, torch.arange(1000))

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
train, test = train_test_split(ds)
train = DataLoader(train, batch_size=args.batchsize)
test = DataLoader(test, batch_size=args.batchsize)
trainer = Trainer(model, optimizer, criterion, train, path)
evaluator = Evaluator(model, metric, test)
evaluator.eval()
trainer.train(args.epochs)
trainer.save()
evaluator.eval()
