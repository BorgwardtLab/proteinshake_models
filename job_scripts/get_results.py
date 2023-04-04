from pathlib import Path
import pandas as pd

OUTPATH = Path("../logs")

def main():
    tasks = [
        'enzyme_class',
        'pfam',
        'ligand_affinity',
        'binding_site_detection',
        'structure_similarity',
        'structural_class',
        'gene_ontology'
    ]

    tasks_property = {
        'enzyme_class': {
            'metric': 'accuracy',
            'name': 'EC'
        },
        'pfam': {
            'metric': 'accuracy',
            'name': 'pfam'
        },
        'ligand_affinity': {
            'metric': 'spearmanr',
            'name': 'LA'
        },
        'binding_site_detection': {
            'metric': 'mcc',
            'name': 'BSD',
        },
        'structure_similarity': {
            'metric': 'spearmanr',
            'name': 'StructSIM'
        },
        'structural_class': {
            'metric': 'accuracy',
            'name': 'StructC'
        }
    }

    splits = ['random', 'sequence', 'structure']
    representations = ['graph', 'point_cloud', 'voxel']
    pretrained_list = ["False", "True"]
    seeds = range(4)

    param_names = ['task', 'split', 'representation', 'pretrained']

    all_results = []
    for task in tasks:
        for split in splits:
            for representation in representations:
                for pretrained in pretrained_list:
                    avg_metric = []
                    for seed in seeds:
                        path = OUTPATH / task / split / representation / pretrained / str(seed) / 'results.csv'
                        try:
                            metric_df = pd.read_csv(path, index_col=0).T
                        except Exception:
                            continue
                        avg_metric.append(metric_df)
                    if len(avg_metric) == 0:
                        continue
                    avg_metric = pd.concat(avg_metric).reset_index(drop=True)
                    metric = pd.DataFrame(avg_metric.mean(axis=0)).T
                    metric_name = tasks_property[task]['metric']
                    metric = metric[[f'test_{metric_name}']]
                    metric[f'{metric_name}_std'] = avg_metric[f'test_{metric_name}'].std()
                    metric['metric'] = metric_name
                    metric['seeds'] = len(avg_metric)
                    params = [task, split, representation, pretrained]
                    for param, param_name in zip(params, param_names):
                        metric[param_name] = [param]
                    # append to big list
                    all_results.append(pd.DataFrame.from_dict(metric))

    table = pd.concat(all_results).reset_index(drop=True)
    print(table.round(6))


if __name__ == "__main__":
    main()
