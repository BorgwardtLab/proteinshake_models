import os
import argparse
import random
import copy
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
from data_utils import PPIDataset
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
    parser.add_argument('--dataset', type=str, default='tm',
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
    parser.add_argument('--pe', type=str, default=None, choices=['None', 'learned', 'sine'])
    parser.add_argument('--out-head', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--aggregation', type=str, default='dot', choices=['dot', 'concat', 'sum'])

    # Optimization hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-06, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--warmup', type=int, default=10, help='warmup epochs')
    parser.add_argument('--scale', action='store_true', help='rescale y')

    # Other hyperparameters
    parser.add_argument('--outdir', type=str, default='../logs', help='out path')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers for loader')
    args = parser.parse_args()

    args.pe = None if args.pe == 'None' else args.pe

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
        if args.dataset in ['ligand_affinity']:
            outdir = outdir + '/{}'.format(args.aggregation)
        os.makedirs(outdir, exist_ok=True)
        args.outdir = outdir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return args

class AttrParser(object):
    def __init__(self, task, y_transform=None):
        self.task = task
        self.y_transform = y_transform

    def __call__(self, data):
        data, protein_dict = data
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        return new_data

class GNNPredictor(pl.LightningModule):
    def __init__(self, model, args, task, y_transform=None):
        super().__init__()
        self.model = model
        self.args = args
        self.task = task
        if task.task_type == 'classification, multi-class':
            self.main_metric = 'acc'
            self.criterion = nn.CrossEntropyLoss()
            self.best_val_score = 0.0
        elif task.task_type == 'regression':
            self.main_metric = 'neg_mse'
            self.criterion = nn.MSELoss()
            self.best_val_score = -float('inf')
        else:
            raise ValueError("Unknown taks type!")
        self.main_val_metric = 'val_' + self.main_metric
        self.best_weights = None
        self.y_transform = y_transform

    def inverse_transform(self, y_true, y_pred):
        if self.y_transform is None:
            return y_true, y_pred
        return self.y_transform.inverse_transform(y_true), self.y_transform.inverse_transform(y_pred)

    def training_step(self, batch, batch_idx):
        data1, data2, y = batch
        y_hat = self.model(data1, data2)
        loss = self.criterion(y_hat, y)

        if 'classification' in self.task.task_type:
            acc = (y_hat.detach().argmax(dim=-1) == y).float().mean().item()
            self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):
        data1, data2, y = batch
        y_hat = self.model(data1, data2)
        loss = self.criterion(y_hat, y)

        self.log('val_loss', loss, batch_size=len(y))
        return {'y_pred': y_hat, 'y_true': y}

    def evaluate_epoch_end(self, outputs, stage='val'):
        all_preds = torch.vstack([out['y_pred'] for out in outputs])
        all_true = torch.cat([out['y_true'] for out in outputs])
        all_preds, all_true = all_true.cpu().numpy(), all_preds.cpu().numpy()
        all_preds, all_true = self.inverse_transform(all_preds, all_true)
        scores = compute_metrics(all_true, all_preds, self.task.task_type)
        scores = {'{}_'.format(stage) + str(key): val for key, val in scores.items()}
        if stage == 'val':
            self.log_dict(scores)
        return scores

    def validation_epoch_end(self, outputs):
        scores = self.evaluate_epoch_end(outputs, 'val')
        if scores[self.main_val_metric] >= self.best_val_score:
            self.best_val_score = scores[self.main_val_metric]
            self.best_weights = copy.deepcopy(self.model.state_dict())
        return scores

    def test_step(self, batch, batch_idx):
        data1, data2, y = batch
        y_hat = self.model(data1, data2)
        loss = self.criterion(y_hat, y)
        return {'y_pred': y_hat, 'y_true': y}

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
        plt.switch_backend("agg")
        import seaborn as sns
        metrics = pd.read_csv(f"{self.logger.log_dir}/metrics.csv")
        del metrics["step"]
        metrics.set_index("epoch", inplace=True)
        if 'classification' in self.task.task_type:
            metric_list = ['val_acc', 'val_loss', 'train_acc', 'train_loss']
        elif 'regression' in self.task.task_type:
            metric_list = ['val_mse', 'val_loss', 'train_loss']
        metrics = metrics[metric_list]
        sns.relplot(data=metrics, kind="line")
        plt.savefig(self.logger.log_dir + '/plot.png')
        plt.close()

def main():
    global args
    args = load_args()
    print(args)

    datapath = "../data/{}".format(args.dataset)
    args.pair_prediction = False
    args.same_type = False
    args.other_dim = None
    if args.dataset == 'tm':
        task = ps_tasks.RetrieveTask(root=datapath)
        dset = task.dataset.to_graph(eps=args.graph_eps).pyg(
            # transform=AttrParser(task)
        )
        num_class = 1
        args.pair_prediction = True
        args.same_type = True
    else:
        raise ValueError("not implemented!")

    # Filter proteins longer than 3000
    protein_len_list = np.asarray([data[0].num_nodes for data in dset])
    print("protein length less or equal to 3000 is {}%".format(
        np.sum(protein_len_list <= 3000) / len(protein_len_list) * 100))
    train_mask = np.all(protein_len_list[task.train_ind] <= 3000, axis=1)
    val_mask = np.all(protein_len_list[task.val_ind] <= 3000, axis=1)
    test_mask = np.all(protein_len_list[task.test_ind] <= 3000, axis=1)

    train_dset = PPIDataset(dset, task, split='train', filter_mask=train_mask, transform=AttrParser(task))
    val_dset = PPIDataset(dset, task, split='val', filter_mask=val_mask, transform=AttrParser(task))
    test_dset = PPIDataset(dset, task, split='test', filter_mask=test_mask, transform=AttrParser(task))
    print(f"Dataset size: train {len(train_dset)}, val {len(val_dset)}, test {len(test_dset)}")

    y_transform = None
    if args.scale:
        from sklearn.preprocessing import StandardScaler
        all_y = np.asarray([y.item() for _,_,y in train_dset])
        scaler = StandardScaler().fit(all_y.reshape(-1, 1))
        train_dset.y_transform = scaler
        val_dset.y_transform = scaler
        test_dset.y_transform = scaler

    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size,
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
        args.out_head,
        args.pair_prediction,
        args.same_type,
        args.other_dim,
        args.aggregation
    )

    if args.pretrained is not None:
        encoder.from_pretrained(args.pretrained + '/model.pt')

    model = GNNPredictor(encoder, args, task, y_transform)

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
    model.model.load_state_dict(model.best_weights)
    model.best_weights = None
    trainer.test(model, test_loader)
    model.plot()
    if args.save_logs:
        encoder.save(args.outdir + '/model.pt', args)


if __name__ == "__main__":
    main()
