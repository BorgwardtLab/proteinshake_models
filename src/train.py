import torch
import yaml
import numpy as np
from src.models.voxel import *
from src.models.point import *
from src.util import *
from proteinshake.datasets import AlphaFoldDataset
from proteinshake.tasks import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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

class PretrainingTask():
    def __init__(self, root):
        self.dataset = AlphaFoldDataset(root=root, organism='swissprot') # methanocaldococcus_jannaschii
        _, size = self.dataset.proteins()
        self.train_ind, self.test_ind = train_test_split(range(size), test_size=0.05)

    def evaluate(self, y_true, y_pred):
        return sklearn.metrics.accuracy_score(y_true, y_pred)

# Dataset
if args.task == 'PT': # pretraining
    task = PretrainingTask(root='data/af')
elif args.task == 'EC':
    task = EnzymeCommissionTask(root='data/ec')
elif args.task == 'LA':
    task = LigandAffinityTask(root='data/la')

# Model
model = {
    'point': {
        'PT': PointNet_Pretrain,
        'EC': PointNet_EC,
        'LA': PointNet_LA
    },
    'voxel': {
        'PT': VoxelNet_Pretrain,
        'EC': VoxelNet_EC,
        'LA': VoxelNet_LA
    }
}[args.representation][args.task](task)

# from pretrained?
if args.pretrained:
    print('Loading pretrained weights')
    model.from_pretrained(f'results/{args.representation}_PT/base_weights.pt')

# create dataset
if args.representation == 'voxel':
    ds = task.dataset.to_voxel(gridsize=(60,35,50), voxelsize=10).torch(transform=model.transform)
if args.representation == 'point':
    ds = task.dataset.to_point().torch(transform=model.transform)

# for testing
#ds = torch.utils.data.Subset(ds, torch.arange(1000))

# setup and run
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6)
#train, test = train_test_split(ds)
train = torch.utils.data.Subset(ds, task.train_ind)
test = torch.utils.data.Subset(ds, task.test_ind)
train = DataLoader(train, batch_size=args.batchsize, shuffle=True)
test = DataLoader(test, batch_size=args.batchsize, shuffle=True)
trainer = Trainer(model, optimizer, train, path)
evaluator = Evaluator(model, test)
evaluator.eval()
trainer.train(args.epochs)
trainer.save()
metrics = evaluator.eval()
with open(path+'/metrics.yml', 'w') as file:
    yaml.dump(metrics, file)
