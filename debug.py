import numpy as np
from proteinshake.datasets import AlphaFoldDataset
from tqdm import tqdm
ds = AlphaFoldDataset(root='data/af', organism='swissprot')

voxelsize = 10
resolution = 'residue'
proteins, size = ds.proteins()
gridsize = np.array([[
    np.ptp(protein[resolution]['x']),
    np.ptp(protein[resolution]['y']),
    np.ptp(protein[resolution]['z'])
    ] for protein in tqdm(proteins, total=size)]).max(0)
gridsize = np.ceil(gridsize/voxelsize).astype(int)
print(gridsize)
