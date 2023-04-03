import os
import argparse
import random
import copy
import numpy as np
import itertools
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset

from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from proteinshake.transforms import Compose

from proteinshake import tasks as ps_tasks
from proteinshake.tasks import __all__ as ALLTASKS

from proteinshake_eval.models.graph import GNN_TYPES
from proteinshake_eval.utils import get_cosine_schedule_with_warmup, get_task
from proteinshake_eval.utils import get_filter_mask
from proteinshake_eval.utils import get_data_loaders
from proteinshake_eval.utils import get_loss
from proteinshake_eval.metrics import compute_metrics
from proteinshake_eval.transforms import get_transformed_dataset
from proteinshake_eval.models.protein_model import ProteinStructureNet

import pytorch_lightning as pl


ALLTASKS = ALLTASKS[1:]


class ProteinTaskTrainer(pl.LightningModule):
    def __init__(self, model, cfg, task, y_transform=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.task = task
        self.criterion, self.main_metric = get_loss(task.task_type[1])
        self.best_val_score = -float('inf')
        self.main_val_metric = 'val_' + self.main_metric
        self.best_weights = None
        self.y_transform = y_transform

    def inverse_transform(self, y_true, y_pred):
        if self.y_transform is None:
            return y_true, y_pred
        return self.y_transform.inverse_transform(y_true), self.y_transform.inverse_transform(y_pred)

    def training_step(self, batch, batch_idx):
        y_hat, y = self.model.step(batch)
        loss = self.criterion(y_hat, y)
        if hasattr(self.model, "regularizer_loss"):
            reg_loss = self.model.regularizer_loss()
            loss = loss + reg_loss

        if 'classification' in self.task.task_type:
            if 'binary' in self.task.task_type:
                acc = ((y_hat.detach() > 0).float() == y).float().mean().item()
                self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
            else:
                acc = (y_hat.detach().argmax(dim=-1) == y).float().mean().item()
                self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.model.step(batch)
        loss = self.criterion(y_hat, y)

        self.log('val_loss', loss, batch_size=len(y))
        return {'y_pred': y_hat, 'y_true': y}

    def evaluate_epoch_end(self, outputs, stage='val'):
        all_preds = torch.vstack([out['y_pred'] for out in outputs])
        all_true = torch.cat([out['y_true'] for out in outputs])
        all_true, all_preds = all_true.cpu().numpy(), all_preds.cpu().numpy()
        all_true, all_preds = self.inverse_transform(all_true, all_preds)
        scores = compute_metrics(all_true, all_preds, self.task)
        # scores = self.task.evaluate(all_true, all_preds)
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
        y_hat, y = self.model.step(batch)
        loss = self.criterion(y_hat, y)
        return {'y_pred': y_hat, 'y_true': y}

    def test_epoch_end(self, outputs):
        scores = self.evaluate_epoch_end(outputs, 'test')
        scores['best_val_score'] = self.best_val_score
        df = pd.DataFrame.from_dict(scores, orient='index')
        df.to_csv(f"{self.logger.log_dir}/results.csv",
                  header=['value'], index_label='name')
        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.cfg.training.warmup, self.cfg.training.epochs
        )
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
            metric_list = ['val_mse', 'val_mae', 'val_loss', 'train_loss']
        metrics = metrics[metric_list]
        sns.relplot(data=metrics, kind="line")
        plt.savefig(self.logger.log_dir + '/plot.png')
        plt.close()


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    task = get_task(cfg.task.class_name)(
        root=cfg.task.path, split=cfg.task.split, verbosity=1)
    dset = task.dataset
    
    # Filter out proteins longer than 3000
    index_masks = get_filter_mask(dset, task, 3000)

    y_transform = None
    if task.task_type[1] == 'regression':
        from sklearn.preprocessing import StandardScaler
        all_y = np.asarray([
            task.target(protein_dict) for protein_dict in dset.proteins()])[task.train_index]
        y_transform = StandardScaler().fit(all_y.reshape(-1, 1))

    dset = get_transformed_dataset(cfg.representation, dset, task, y_transform, max_len=3000)
    if "pair" in task.task_type[0] or cfg.task.name == 'ligand_affinity':
        task.pair_data = True
    else:
        task.pair_data = False
    task.other_dim = dset[0].other_x.shape[-1] if cfg.task.name == 'ligand_affinity' else None 
    net = ProteinStructureNet(cfg.model, task)

    train_loader, val_loader, test_loader = get_data_loaders(
        dset, task, index_masks,
        cfg.training.batch_size, cfg.training.num_workers
    )

    if cfg.model.pretrained is not None:
        print("Loading pretrained model...")
        net.from_pretrained(cfg.model.pretrained + '/model.pt')

    model = ProteinTaskTrainer(net, cfg, task, y_transform)

    logger = pl.loggers.CSVLogger(cfg.paths.output_dir, name='csv_logs')
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=1000)
    ]

    limit_train_batches = 5 if cfg.training.debug else None
    limit_val_batches = 5 if cfg.training.debug else None
    # enable_checkpointing = False if args.debug else True
    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=cfg.training.epochs,
        devices='auto',
        accelerator='auto',
        enable_checkpointing=False,
        # default_root_dir=args.outdir,
        logger=[logger],
        callbacks=callbacks
    )

    trainer.fit(model, train_loader, val_loader)
    model.model.load_state_dict(model.best_weights)
    model.best_weights = None
    trainer.test(model, test_loader)
    net.save(f"{cfg.paths.output_dir}/model.pt")
    # model.plot()


if __name__ == "__main__":
    main()
