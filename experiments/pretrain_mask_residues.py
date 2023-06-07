import os
import random
import numpy as np
import itertools
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric import utils
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from proteinshake import datasets

from proteinshake_eval.transforms import get_pretrain_dataset
from proteinshake_eval.models.protein_model import ProteinStructureEncoder

from proteinshake_eval.utils import get_cosine_schedule_with_warmup

import pytorch_lightning as pl
import logging

log = logging.getLogger(__name__)


class MaskTrainer(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        y_hat, y = self.model.step(batch)
        loss = self.criterion(y_hat, y)
        if hasattr(self.model.encoder, "regularizer_loss"):
            reg_loss = self.model.encoder.regularizer_loss()
            loss = loss + reg_loss

        acc = (y_hat.detach().argmax(dim=-1) == y).float().mean().item()
        self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1)

        return loss

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

@hydra.main(version_base="1.3", config_path="../config", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    dset = datasets.AlphaFoldDataset(root=cfg.task.path, organism=cfg.task.organism)

    dset = get_pretrain_dataset(cfg.representation, dset)

    data_loader = DataLoader(
        dset, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers
    )

    net = ProteinStructureEncoder(cfg.model)

    model = MaskTrainer(net, cfg)

    logger = pl.loggers.CSVLogger(cfg.paths.output_dir, name='csv_logs')
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=1000)
    ]

    limit_train_batches = 5 if cfg.training.debug else None

    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        max_epochs=cfg.training.epochs,
        devices='auto',
        accelerator='auto',
        enable_checkpointing=False,
        logger=[logger],
        callbacks=callbacks
    )

    trainer.fit(model=model, train_dataloaders=data_loader)

    save_dir = Path(cfg.paths.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    net.save(save_dir / "model.pt")


if __name__ == "__main__":
    main()
