import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D


df = pd.read_csv('results.csv')
column_map = {'task':'Task','split':'Split','representation':'Representation','pretrained':'Pre-trained'}
task_map = {'enzyme_class':'Enzyme Class', 'pfam':'Protein Family', 'ligand_affinity': 'Ligand Affinity', 'binding_site_detection': 'Binding Site', 'structure_similarity': 'Structure Similarity', 'structural_class': 'Structural Class', 'gene_ontology':'Gene Ontology', 'protein_protein_interface':'Protein-Protein Interface'}
task_map_short = {'Enzyme Class':'EC', 'Protein Family':'PF', 'Ligand Affinity': 'LA', 'Binding Site': 'BS', 'Structure Similarity': 'SS', 'Structural Class': 'SC', 'Gene Ontology':'GO', 'Protein-Protein Interface':'PPI'}
split_map = {'random':'Random', 'sequence':'Sequence', 'structure':'Structure'}
split_map_short = {'Random':'Rnd', 'Sequence':'Seq', 'Structure':'Str'}
rep_map = {'point_cloud':'Point', 'voxel':'Voxel', 'graph':'Graph'}
dfs = []
for seed in range(4):
    d = df[[f'test_score_{seed}','metric','task','split','representation','pretrained']].rename(columns={f'test_score_{seed}':'Score',**column_map})
    d['Seed'] = seed
    dfs.append(d)
df = pd.concat(dfs)
df.Task = df.Task.map(task_map)
df.Split = df.Split.map(split_map)
df.Representation = df.Representation.map(rep_map)

# 1 (one whole table): Representations. Table of Representations vs. Task vs. Split
table = df[~df['Pre-trained']]
table.loc[:,'Split'] = table['Split'].map(split_map_short)
mean = table.groupby(['Task','Representation','Split']).agg({'Score':'mean'}).unstack(level=1)
mean.columns = mean.columns.droplevel(0)
max_indices = mean.idxmax(axis=1)
mask = mean.applymap(lambda x: False)
for i,v in max_indices.items(): mask.loc[i,v] = True
table = mean.applymap('{:.3f}'.format).astype(str)
table = table.mask(mask, lambda x: '\\textbf{'+x+'}')
table = table.unstack(level=1)
#table = table.swaplevel(0, 1, axis=1)
#table.sort_index(axis=1, level=0, inplace=True)
table = table.to_latex(escape=False).replace('nan','-').replace('llll','lccc')
with open(f'1_Representation.txt','w') as file:
        file.write(table)
'''

'''
# 1: Representations. Table of Representations vs. Task, per Split
data = df[~df['Pre-trained']]
for split in df.Split.unique():
    table = data[data.Split == split]
    mean = table.groupby(['Task','Representation']).agg({'Score':'mean'}).unstack(level=1)
    std = table.groupby(['Task','Representation']).agg({'Score':'std'}).unstack(level=1)
    mean.columns = mean.columns.droplevel(0)
    std.columns = std.columns.droplevel(0)
    max_indices = mean.idxmax(axis=1)
    mask = mean.applymap(lambda x: False)
    for i,v in max_indices.items(): mask.loc[i,v] = True
    table = mean.applymap('{:.3f}'.format).astype(str) + ' $\pm$ ' + std.applymap('{:.3f}'.format).astype(str)

    markdown = table.mask(mask, lambda x: '**'+x+'**').to_markdown().replace('nan $\pm$ nan','-')
    with open(f'1_Representation_{split}.md','w') as file:
        file.write(markdown)
    
    latex = table.mask(mask, lambda x: '\\textbf{'+x+'}').to_latex(escape=False).replace('nan $\pm$ nan','-').replace('llll','lccc')
    with open(f'1_Representation_{split}.tex','w') as file:
        file.write(latex)


# 2: Splits. Barplot, Supergroup: Representation, Group: Split, per Pretraining
fig, axes = plt.subplots(2,4, figsize=(10,5))
data = df[~df['Pre-trained']]
data['metric'] = data['metric'].map({'accuracy':'Accuracy', 'spearmanr':'Spearman R', 'mcc':'MCC', 'Fmax':'$F_{max}$', 'auroc_median':'AUROC (median)'})
#data.loc[:,'Task'] = data['Task'].map(task_map_short)
for ax,task in zip(axes.flatten(), df.Task.unique()):
    task_data = data[data.Task == task]
    if len(task_data) == 0: continue
    ax.title.set_text(task)
    sns.barplot(data=task_data, x='Representation', y='Score', hue='Split', ax=ax, palette=['lightsalmon','darkseagreen','skyblue'])
    ax.get_legend().remove()
    ax.set_ylabel(task_data.metric.to_list()[0])
    ax.set_xlabel('')

custom_lines = [Line2D([0], [0], color='lightsalmon', lw=6),
                Line2D([0], [0], color='darkseagreen', lw=6),
                Line2D([0], [0], color='skyblue', lw=6)]
axes[0,2].legend(custom_lines, ['Random Split','Sequence Split','Structure Split'], bbox_to_anchor=(1.3,1.5), ncols=3, handlelength=0.5, frameon=False)
plt.subplots_adjust(top=0.85, hspace=0.5, wspace=0.5)
plt.savefig(f'2_Splits.svg')
plt.close()


# 3: Pretraining (alt). Barplot, relative improvement, Supergroup: Task, Group: Split, Subplot: Rep
fig, axes = plt.subplots(1,3, figsize=(10,3))
pt, no_pt = df[df['Pre-trained']], df[~df['Pre-trained']]
data = pd.concat([no_pt,pt], ignore_index=True)
data['Task'] = data['Task'].map(task_map_short)
data['Difference'] = data.groupby(['Task','Representation','Split','Seed'])['Score'].diff()
no_pt = data[~data['Pre-trained']]
data = data[data['Pre-trained']]
data['Improvement [%]'] = data['Difference'].to_numpy() / no_pt['Score'].to_numpy() * 100
data = data[data['Split']=='Random']
for ax,rep in zip(axes.flatten(), df.Representation.unique()):
    rep_data = data[(data.Representation == rep)]
    order = rep_data.groupby('Task')['Improvement [%]'].mean().sort_values().index[::-1]
    rep_data.Task = rep_data.Task.astype("category")
    rep_data.Task = rep_data.Task.cat.set_categories(order)
    ax.title.set_text(rep)
    boxes = sns.pointplot(data=rep_data, y='Task', x='Improvement [%]', ax=ax, join=False, errorbar='sd', color='lightsteelblue')
    ax.axvline(0, color='lightcoral', dashes=(3,3))
    ax.grid()
    lim = rep_data['Improvement [%]'].max()
    ax.set_xlim((-lim,lim))
    for path in ax.collections:
        path.set(color='steelblue', zorder=10)
plt.tight_layout()
plt.savefig(f'3_Pretraining.svg')
plt.close()

# Leaderboard: json format
import json
reverse_task_map = {v:k for k,v in task_map.items()}
data = df[~df['Pre-trained']]
for task in df.Task.unique():
    task_data = data[data.Task == task]
    json_string = []
    for rep in task_data.Representation.unique():
        rep_data = task_data[task_data.Representation == rep]
        json_string.append({
            "Name":"ProteinShake Baseline",
            "Author": "Kucera et al. 2023",
            "Paper": "https://github.com/BorgwardtLab/proteinshake",
            "Code": "https://github.com/BorgwardtLab/ProteinShake_eval",
            "Representation": rep,
            "Random Split": '{:.3f}'.format(rep_data[rep_data.Split == 'Random']['Score'].mean()),
            "Sequence Split": '{:.3f}'.format(rep_data[rep_data.Split == 'Sequence']['Score'].mean()),
            "Structure Split": '{:.3f}'.format(rep_data[rep_data.Split == 'Structure']['Score'].mean())
        })
    with open(f'json/{reverse_task_map[task]}.json','w') as file:
        file.write(json.dumps(json_string, indent=4))