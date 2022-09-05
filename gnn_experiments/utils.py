import math
import torch


class OneHotToIndex(object):
    def __call__(self, data):
        data.x = data.x.argmax(dim=-1)
        return data

class MaskNode(object):
    def __init__(self, num_node_types, mask_rate=0.15):
        self.num_node_types = num_node_types
        self.mask_rate = mask_rate

    def __call__(self, data):
        num_nodes = data.num_nodes
        subset_mask = torch.rand(num_nodes) < self.mask_rate

        data.masked_node_indices = subset_mask
        data.masked_node_label = data.x[subset_mask]
        # data.x = data.x.clone()
        data.x[subset_mask] = self.num_node_types

        return data

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
