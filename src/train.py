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

# Path
path = f'results/{args.representation}_{args.task}'
if args.pretrained:
    path += '_pretrained'

# Dataset
if args.task == 'PT': # pretraining
    ds = AlphaFoldDataset(root='data/af', organism='swissprot') # methanocaldococcus_jannaschii
elif args.task == 'EC':
    ds = EnzymeCommissionDataset(root='data/ec')

# Model
model = {
    'point': {
        'PT': PointNet_Pretrain,
        'EC': PointNet_EC,
    },
    'voxel': {
        'PT': VoxelNet_Pretrain,
        'EC': VoxelNet_EC,
    }
}[args.representation][args.task]()

# from pretrained?
if args.pretrained:
    print('Loading pretrained weights')
    model.from_pretrained(f'results/{args.representation}_PT/base_weights.pt')

# create dataset
if args.representation == 'voxel':
    ds = ds.to_voxel(gridsize=(60,35,50), voxelsize=10).torch(transform=model.transform)
if args.representation == 'point':
    ds = ds.to_point().torch(transform=model.transform)

# for testing
#ds = torch.utils.data.Subset(ds, torch.arange(1000))

# setup and run
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6)
train, test = train_test_split(ds)
train = DataLoader(train, batch_size=args.batchsize, shuffle=True)
test = DataLoader(test, batch_size=args.batchsize, shuffle=True)
trainer = Trainer(model, optimizer, model.criterion, train, path)
evaluator = Evaluator(model, model.metric, test)
evaluator.eval()
trainer.train(args.epochs)
trainer.save()
evaluator.eval()
