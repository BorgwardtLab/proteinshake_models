import os
import argparse
import random
import numpy as np
import itertools
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset

from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from proteinshake import tasks as ps_tasks

from models import GNN, GNN_graphpred, NodeClassifier, GNN_TYPES
from utils import ResidueIdx
from utils import get_cosine_schedule_with_warmup
from metrics import compute_metrics

import pytorch_lightning as pl


def load_args():
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of GNNs on Atom3d',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default='ec',
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
    parser.add_argument('--out-head', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')

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
        if args.pretrained is None:
            outdir = args.outdir + '/{}'.format(args.dataset)
            outdir = outdir + '/{}_{}'.format(args.lr, args.weight_decay)
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.pooling, args.out_head, args.gnn_type, args.num_layers,
                args.embed_dim, args.dropout, args.use_edge_attr, args.pe
            )
        else:
            outdir = args.pretrained + '/{}'.format(args.dataset)
            outdir = outdir + '/{}_{}'.format(args.lr, args.weight_decay)
            outdir = outdir + '/{}_{}_{}'.format(
                args.pooling, args.out_head, args.dropout
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
    def __init__(self, task):
        self.task = task

    def __call__(self, data, protein_dict):
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        new_data.y = self.task.target(protein_dict)
        return new_data

class GNNPredictor(pl.LightningModule):
    def __init__(self, model, args, task):
        super().__init__()
        self.model = model
        self.args = args
        self.task = task
        if args.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif args.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("unknown taks type!")

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch)

        loss = self.criterion(y_hat, batch.y)

        acc = (y_hat.detach().argmax(dim=-1) == batch.y).float().mean().item()
        self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch)
        loss = self.criterion(y_hat, batch.y)

        self.log('val_loss', loss, batch_size=len(batch.y))
        return {'y_pred': y_hat, 'y_true': batch.y}

    def evaluate_epoch_end(self, outputs, stage='val'):
        all_preds = torch.vstack([out['y_pred'] for out in outputs])
        all_true = torch.cat([out['y_true'] for out in outputs])
        scores = compute_metrics(all_true.numpy(), all_preds.numpy())
        scores = {'{}_'.format(stage) + str(key): val for key, val in scores.items()}
        if stage == 'val':
            self.log_dict(scores)
        return scores

    def validation_epoch_end(self, outputs):
        return self.evaluate_epoch_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch)
        loss = self.criterion(y_hat, batch.y)
        return {'y_pred': y_hat, 'y_true': batch.y}

    def test_epoch_end(self, outputs):
        scores = self.evaluate_epoch_end(outputs, 'test')
        df = pd.DataFrame.from_dict(scores, orient='index')
        df.to_csv(self.logger.log_dir + '/results.csv',
                  header=['value'], index_label='name')
        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, self.args.warmup, self.args.epochs)
        return [optimizer], [lr_scheduler]

    def plot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        metrics = pd.read_csv(f"{self.logger.log_dir}/metrics.csv")
        del metrics["step"]
        metrics.set_index("epoch", inplace=True)
        metrics = metrics[['val_acc', 'val_loss', 'train_acc', 'train_loss']]
        sns.relplot(data=metrics, kind="line")
        plt.savefig(self.logger.log_dir + '/plot.png')
        plt.close()

def main():
    global args
    args = load_args()
    print(args)

    datapath = "../data/{}".format(args.dataset)
    if args.dataset == 'ec':
        task = ps_tasks.EnzymeCommissionTask(root=datapath)
        dset = task.dataset.to_graph(eps=args.graph_eps).pyg(
            transform=AttrParser(task)
        )
        num_class = task.num_classes
        args.task_type = 'classification'
    else:
        raise ValueError("not implemented!")

    train_loader = DataLoader(Subset(dset, task.train_ind), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(Subset(dset, task.val_ind), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(Subset(dset, task.test_ind), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    encoder = GNN_graphpred(
        num_class,
        args.embed_dim,
        args.num_layers,
        args.dropout,
        args.gnn_type,
        args.use_edge_attr,
        args.pe,
        args.pooling,
        args.out_head
    )

    if args.pretrained is not None:
        encoder.from_pretrained(args.pretrained + '/model.pt')

    model = GNNPredictor(encoder, args, task)

    logger = pl.loggers.CSVLogger(args.outdir, name='csv_logs')
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=1000)
    ]

    limit_train_batches = 5 if args.debug else None
    limit_val_batches = 5 if args.debug else None
    # enable_checkpointing = False if args.debug else True
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
    trainer.test(model, test_loader)
    model.plot()
    if args.save_logs:
        encoder.save(args.outdir + '/model.pt', args)


if __name__ == "__main__":
    main()
