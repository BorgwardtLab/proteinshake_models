import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch

def plot_voxel(voxel, num):

    nonzero = ~((voxel == 0).all(-1))
    def mask():
        volume = nonzero.sum()
        n, m = int(volume * 0.15), volume-int(volume * 0.15)
        mask = torch.zeros(voxel.shape[:-1]).bool()
        inner_mask = torch.cat([torch.ones(n),torch.zeros(m)])[torch.randperm(volume)].bool()
        mask[nonzero] = inner_mask
        return mask.numpy()
    cmap = matplotlib.cm.get_cmap('hsv')
    x, y, z = np.indices(np.array(nonzero.shape)+1)
    #colors[nonzero] = np.array([0.0,0.5,0.5])
    #colors[mask()] = np.array([0.5,0.0,0.5])
    #nonzero = nonzero.numpy()


    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x,y,z,nonzero, edgecolor='k')
    plt.savefig(f'results/plots/voxel_{num}.png')
    plt.close()

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x,y,z,mask(), edgecolor='k')
    plt.savefig(f'results/plots/mask_{num}.png')
    plt.close()

if __name__ == '__main__':
    from proteinshake.datasets import AlphaFoldDataset
    ds = AlphaFoldDataset(root='data/plot', organism='methanocaldococcus_jannaschii').to_voxel().torch()
    for num in range(5):
        plot_voxel(ds[num][0], num)
