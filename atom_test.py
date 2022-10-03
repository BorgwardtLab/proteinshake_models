import tempfile

from datasets.atom3d_data import Atom3DDataset
from torch_geometric.data import DataLoader

# with tempfile.TemporaryDirectory() as td:
tasks = ('lba', 'psr', 'msp', 'ppi', 'res', 'lep')
for task in tasks:
    proteins = Atom3DDataset(root=task, atom_dataset=task, use_precomputed=False)
    proteins = proteins.to_graph(k=5).pyg()
    dl = DataLoader(proteins, batch_size=2)

