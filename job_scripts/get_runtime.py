import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from proteinshake_eval.utils import get_task, get_filter_mask
from proteinshake_eval.utils import get_data_loaders
from proteinshake_eval.transforms import get_transformed_dataset
from proteinshake_eval.models.protein_model import ProteinStructureNet
from torch.profiler import profile, record_function, ProfilerActivity

from timeit import default_timer as timer


@hydra.main(version_base="1.3", config_path="../config", config_name="profiler")
def main(cfg: DictConfig) -> None:

    task = get_task(cfg.task.class_name)(
        root=cfg.task.path, split=cfg.task.split, verbosity=1)
    dset = task.dataset
    
    # Filter out proteins longer than 3000
    max_len = 3000
    if task.task_type[0] == 'residue_pair':
        max_len //= 2
    index_masks = get_filter_mask(dset, task, max_len)

    y_transform = None
    if task.task_type[1] == 'regression':
        from sklearn.preprocessing import StandardScaler
        task.compute_targets()
        all_y = task.train_targets
        y_transform = StandardScaler().fit(all_y.reshape(-1, 1))

    dset = get_transformed_dataset(cfg.representation, dset, task, y_transform)
    if "pair" in task.task_type[0] or cfg.task.name == 'ligand_affinity':
        task.pair_data = True
    else:
        task.pair_data = False
    task.other_dim = dset[0].other_x.shape[-1] if cfg.task.name == 'ligand_affinity' else None 
    net = ProteinStructureNet(cfg.model, task)
    if torch.cuda.is_available():
        net.to("cuda")

    train_loader, val_loader, test_loader = get_data_loaders(
        dset, task, index_masks,
        cfg.training.batch_size, cfg.training.num_workers
    )

    if cfg.pretrained:
        print("Loading pretrained model...")
        net.from_pretrained(cfg.pretrained_path)

    runtime = 0.
    n_runs = 30
    for i, data in enumerate(test_loader):
        if i == n_runs:
            break
        if torch.cuda.is_available():
            data = data.to("cuda")
            torch.cuda.synchronize()
        tic = timer()
        y, y_hat = net.val_step(data)
        toc = timer()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime += toc - tic

    ppps = cfg.training.batch_size * n_runs / runtime
    print(f"number of proteins processed per second: {ppps:.2f}")

    train_loader, val_loader, test_loader = get_data_loaders(
        dset, task, index_masks,
        32, 0
    )
    for inputs in test_loader:
        break

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            net.val_step(inputs)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



if __name__ == "__main__":
    main()
