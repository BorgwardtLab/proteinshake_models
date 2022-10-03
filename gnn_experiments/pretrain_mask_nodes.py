import os
import argparse
import random
import numpy as np
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric import utils
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from proteinshake import datasets

from models import GNN, NodeClassifier, GNN_TYPES
from utils import OneHotToIndex, MaskNode, ResidueIdx
from utils import get_cosine_schedule_with_warmup

import pytorch_lightning as pl


def load_args():
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training on AlphaFold2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--organism', type=str, default='swissprot',
                        help='which organism for pre-training')
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
    parser.add_argument('--pe', type=str, default=None, choices=['learned', 'sine'])
    parser.add_argument('--mask-rate', type=float, default=0.15,
                        help='masking ratio')

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
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for loader')
    args = parser.parse_args()

    if args.debug:
        args.organism = 'escherichia_coli'
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
            args.mask_rate, args.gnn_type, args.num_layers,
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

class GNNMasking(pl.LightningModule):
    def __init__(self, model, classifier, args):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        node_true = batch.masked_node_label
        node_repr = self.model(batch)
        node_pred = self.classifier(node_repr[batch.masked_node_indices])

        loss = self.criterion(node_pred, node_true)

        acc = (node_pred.detach().argmax(dim=-1) == node_true).float().mean().item()
        self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.model.parameters(), self.classifier.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, self.args.warmup, self.args.epochs)
        return [optimizer], [lr_scheduler]


def main():
    global args
    args = load_args()
    print(args)

    datapath = '../data/AlphaFold/{}'.format(args.organism)
    dset = datasets.AlphaFoldDataset(
        root=datapath, organism=args.organism).to_graph(eps=args.graph_eps).pyg(
        transform=T.Compose([ResidueIdx(), MaskNode(20, mask_rate=args.mask_rate)])
    )

    data_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = GNN(
        args.embed_dim,
        args.num_layers,
        args.dropout,
        args.gnn_type,
        args.use_edge_attr,
        args.pe
    )

    classifier = NodeClassifier(args.embed_dim)

    model = GNNMasking(encoder, classifier, args)

    logger = pl.loggers.CSVLogger(args.outdir, name='csv_logs')
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=1000)
    ]

    limit_train_batches = 5 if args.debug else None

    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        max_epochs=args.epochs,
        devices='auto',
        accelerator='auto',
        enable_checkpointing=False,
        default_root_dir=args.outdir,
        logger=[logger],
        callbacks=callbacks
    )

    trainer.fit(model=model, train_dataloaders=data_loader)

    if args.save_logs:
        encoder.save(args.outdir + '/model.pt', args)

    # sanity check for node masking
    # for data in data_loader:
    #     print(data)
    #     print(data.x.shape)
    #     print(data.x)
    #     print(data.masked_node_label)
    #     print(data.masked_node_indices)
    #     # out = model(data)
    #     # print(out)
    #     # print(out.shape)
    #     break

    # for data in data_loader:
    #     print(data)
    #     print(data.x.shape)
    #     print(data.x)
    #     print(data.masked_node_label)
    #     print(data.masked_node_indices)
    #     # out = model(data)
    #     # print(out)
    #     # print(out.shape)
    #     break




if __name__ == "__main__":
    main()
