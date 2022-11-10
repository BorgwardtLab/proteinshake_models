

class PointMaskingTransform():

    def __call__(self, data, protein_dict, mask_ratio=0.15):
        coords, labels = data[:,:3], data[:,3]
        labels = torch.eye(20)[labels.long()].float()
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        length = labels.shape[0]
        n, m = int(length * mask_ratio), length-int(length * mask_ratio)
        mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(length)].bool()
        masked = labels.clone()
        masked[mask] = torch.ones(labels.shape[1]).float()
        return coords, labels, masked, mask


class PointEnzymeClassTransform():

    def __init__(self, task):
        self.task = task

    def __call__(self, data, protein_dict):
        coords, labels = data[:,:3], data[:,3]
        labels = torch.eye(20)[labels.long()].float()
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        ec = torch.eye(self.task.num_classes)[self.task.target(protein_dict)].float()
        return coords, labels, ec


class PointLigandAffinityTransform():

    def __init__(self, task):
        self.task = task

    def transform(self, data, protein_dict):
        coords, labels = data[:,:3], data[:,3]
        labels = torch.eye(20)[labels.long()].float()
        L = 1024
        coords = torch.nn.functional.pad(coords[:L], (0,0,0,max(0,L-coords.shape[0])))
        labels = torch.nn.functional.pad(labels[:L], (0,0,0,max(0,L-labels.shape[0])))
        la = torch.tensor(self.task.target(protein_dict)).float()
        return coords, labels, la
