import math
import torch
from torch_geometric.data import Data


class OneHotToIndex(object):
    def __call__(self, data):
        data.x = data.x.argmax(dim=-1)
        return data

class ResidueIdx(object):
    def __call__(self, data):
        data.residue_idx = torch.arange(data.num_nodes)
        return data



def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
