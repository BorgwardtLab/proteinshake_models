import torch, os, yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer():

    def __init__(self, model, optimizer, train, test, val, path, verbose=True):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.criterion = model.criterion
        self.dataloader = train
        self.path = path
        self.verbose = verbose
        self.losses = []
        self.test = test
        self.val = val
        os.makedirs(path, exist_ok=True)

    def train(self, epochs):
        self.model.train()
        epoch_pbar = tqdm(range(epochs), desc='Epoch') if self.verbose else range(epochs)
        for epoch in epoch_pbar:
            batch_pbar = tqdm(self.dataloader, desc='Batch', leave=False) if self.verbose else self.dataloader
            for batch in batch_pbar:
                loss = self.train_step(batch)
                self.losses.append(loss)
            self.plot()
            self.eval('val')
        metrics = self.eval('test')
        with open(self.path+'/metrics.yml', 'w') as file:
            yaml.dump(metrics, file)

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

    def eval(self, dataloader):
        self.model.eval()
        dataloader = self.test if dataloader == 'test' else self.val
        batch_pbar = tqdm(dataloader, desc='Batch', leave=False) if self.verbose else dataloader
        y_true, y_pred = [],[]
        for batch in batch_pbar:
            _y_true, _y_pred = self.model.predict(batch)
            y_true.extend(_y_true.cpu().detach())
            y_pred.extend(_y_pred.cpu().detach())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        #print(y_true,y_pred)
        #print(y_true.shape, y_pred.shape)
        metrics = self.model.task.evaluate(y_true, y_pred)
        print(metrics)
        self.model.train()
        return {k:float(v) for k,v in metrics.items()}


def train_test_split(ds):
    return torch.utils.data.random_split(ds, [int(np.floor(len(ds)*0.9)), int(np.ceil(len(ds)*0.1))], generator=torch.Generator().manual_seed(0))
