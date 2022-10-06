import torch, os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer():

    def __init__(self, model, optimizer, criterion, dataloader, path, verbose=True):
        self.model = model.train().cuda()
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.path = path
        self.verbose = verbose
        self.losses = []
        os.makedirs(path, exist_ok=True)

    def train(self, epochs):
        epoch_pbar = tqdm(range(epochs), desc='Epoch') if self.verbose else range(epochs)
        for epoch in epoch_pbar:
            batch_pbar = tqdm(self.dataloader, desc='Batch', leave=False) if self.verbose else self.dataloader
            for batch in batch_pbar:
                loss = self.train_step(batch)
                self.losses.append(loss)
            self.plot()

    def train_step(self, batch):
        self.optimizer.zero_grad()
        output = self.model(batch)
        loss = self.criterion(batch, output)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def save(self):
        self.model.save_weights(self.path+'/weights.pt')
        self.model.base.save_weights(self.path+'/base_weights.pt')

    def plot(self):
        sns.lineplot(x=np.arange(len(self.losses)), y=self.losses)
        plt.savefig(self.path+'/loss.png')
        plt.close()

class Evaluator():

    def __init__(self, model, metric, dataloader, verbose=True):
        self.model = model.eval().cuda()
        self.metric = metric
        self.dataloader = dataloader
        self.verbose = verbose
        self.metrics = []

    def eval(self):
        batch_pbar = tqdm(self.dataloader, desc='Batch', leave=False) if self.verbose else self.dataloader
        for batch in batch_pbar:
            metric = self.eval_step(batch)
            self.metrics.append(metric)
        print(np.mean(self.metrics))

    def eval_step(self, batch):
        output = self.model(batch)
        m = self.metric(batch, output)
        return float(m)


def train_test_split(ds):
    return torch.utils.data.random_split(ds, [int(np.floor(len(ds)*0.9)), int(np.ceil(len(ds)*0.1))], generator=torch.Generator().manual_seed(0))
