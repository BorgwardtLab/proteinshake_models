import os
import argparse
import random
import numpy as np
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from datasets.atom3d_data import Atom3DDataset

from models import GNN, GNN_graphpred, NodeClassifier, GNN_TYPES
from utils import ResidueIdx
from utils import get_cosine_schedule_with_warmup

import pytorch_lightning as pl


def load_args():
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of GNNs on Atom3d',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default='psr',
                        help='which dataset')
    parser.add_argument('--graph-eps', type=float, default=8.0,
                        help='constructing eps graphs from distance matrices')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode using escherichia_coli subset')

    # Model hyperparameters
    parser.add_argument('--num-layers', type=int, default=5, help="number of layers")
    parser.add_argument('--embed-dim', type=int, default=256, help="hidden dimensions")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout")
    parser.add_argument('--gnn-type', type=str, default='gin', choices=GNN_TYPES,
                        help='gnn type')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--pooling', type=str, default='mean', help='global pooling')
    parser.add_argument('--pe', type=str, default=None, choices=['learned', 'sine'])

    # Optimization hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-06, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--warmup', type=int, default=10, help='warmup epochs')

    # Other hyperparameters
    parser.add_argument('--outdir', type=str, default='../logs', help='out path')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers for loader')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 10
        args.embed_dim = 16
        args.num_layers = 2
        args.outdir = '../logs_debug'

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        outdir = outdir + '/{}_{}'.format(args.lr, args.weight_decay)
        outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}'.format(
            args.pooling, args.gnn_type, args.num_layers,
            args.embed_dim, args.dropout, args.use_edge_attr, args.pe
        )
        os.makedirs(outdir, exist_ok=True)
        args.outdir = outdir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return args


class AttrParser(object):
    def __call__(self, data):
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = data.residue_number - 1
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        new_data.y = data.gdt_ha
        return new_data


class GNNPredictor(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        print(batch_idx)
        y_hat = self.model(batch)

        loss = self.criterion(y_hat, batch.y)

        # acc = (node_pred.detach().argmax(dim=-1) == node_true).float().mean().item()
        # self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self.model(batch)
        loss = self.criterion(y_pred, batch.y)

        self.log('val_loss', loss, batch_size=len(batch.y))
        return y_pred

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.stack(validation_step_outputs)
        print(all_preds)

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch)
        loss = self.criterion(y_hat, batch.y, batch_size=len(batch.y))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, self.args.warmup, self.args.epochs)
        return [optimizer], [lr_scheduler]

def main():
    global args
    args = load_args()
    print(args)

    datapath = "../data/atom3d/{}".format(args.dataset)
    dset = Atom3DDataset(
        root=datapath, atom_dataset=args.dataset).to_graph(eps=args.graph_eps).pyg(
        transform=T.Compose([AttrParser()])
    )

    # splits_path = '../splits/atom3d/{}/year/{}_indices.txt'
    # splits = {}
    # for split in ['train', 'val', 'test']:
    #     splits[split] = torch.from_numpy(np.loadtxt(splits_path.format(args.dataset, split), dtype=np.int64))
    #     print(splits[split].max())
    #     print(len(splits[split]))
    # print(splits)
    # print(len(dset))
    # print(dset[186358])
    # print(len(dset[splits['train']]))


    train_loader = DataLoader(dset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # for data in train_loader:
    #     print(data)
    #     print(data.y)
    #     # print(data.affinity)
    #     return

    # return

    num_class = 1
    encoder = GNN_graphpred(
        num_class,
        args.embed_dim,
        args.num_layers,
        args.dropout,
        args.gnn_type,
        args.use_edge_attr,
        args.pe,
        args.pooling,
    )

    model = GNNPredictor(encoder, args)

    logger = pl.loggers.CSVLogger(args.outdir, name='csv_logs')
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=1000)
    ]

    limit_train_batches = 5 if args.debug else None
    limit_val_batches = 5 if args.debug else None
    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=args.epochs,
        devices='auto',
        accelerator='auto',
        enable_checkpointing=False,
        default_root_dir=args.outdir,
        logger=[logger],
        callbacks=callbacks
    )

    trainer.fit(model, train_loader, val_loader)




if __name__ == "__main__":
    main()
