"""600-dpi optional publication-quality figures; no empirical values invented."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch

def training_curves(log_file,output,dpi=600):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    log=pd.read_json(log_file,lines=True);metrics=[x for x in ['g','d','validation_med_fid','train_generator_loss','train_loss'] if x in log]
    if not metrics:raise ValueError('No plottable training curves in JSONL')
    fig,ax=plt.subplots(figsize=(7,4.5));
    for metric in metrics:ax.plot(log['epoch'],log[metric],label=metric,marker='o',markersize=2)
    ax.set_xlabel('Epoch');ax.set_ylabel('Metric value');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=dpi);plt.close(fig)

def candidate_grid(manifest,output,max_items=16,dpi=600):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.read_csv(manifest)
    if 'is_synthetic' in df:df=df[df['is_synthetic'].astype(str).str.lower().isin(['true','1'])]
    if df.empty:raise ValueError('No synthetic images to plot')
    df=df.head(max_items)
    ncols=4;nrows=int(np.ceil(len(df)/ncols));fig,axs=plt.subplots(nrows,ncols,figsize=(8,2*nrows),squeeze=False)
    for ax in axs.ravel():ax.axis('off')
    for ax,(_,row) in zip(axs.ravel(),df.iterrows()):
        with Image.open(row.path) as im:ax.imshow(im.convert('L'),cmap='gray',vmin=0,vmax=255)
        ax.set_xlabel(f"class {row['y']} / view {row['view']}",fontsize=7)
    fig.tight_layout();Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=dpi);plt.close(fig)

def class_balance_plot(balanced_manifest,output,dpi=600):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.read_csv(balanced_manifest)
    data=df[df['split']=='train'].copy()
    data['kind']=np.where(data['is_synthetic'].astype(str).str.lower().isin(['true','1']),'Synthetic','Real')
    counts=data.groupby(['y','view','kind']).size().unstack(fill_value=0)
    if 'Real' not in counts:counts['Real']=0
    if 'Synthetic' not in counts:counts['Synthetic']=0
    labels=[f'{i}:{j}' for i,j in counts.index]
    inds=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(max(6,len(labels)*.45),4.2))
    ax.bar(inds,counts['Real'],label='Real')
    ax.bar(inds,counts['Synthetic'],bottom=counts['Real'],label='Accepted synthetic')
    ax.set_xticks(inds,labels,rotation=45,ha='right');ax.set_xlabel('Disease class : view ID')
    ax.set_ylabel('Training images');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=dpi);plt.close(fig)


def confusion_matrix_plot(prediction_csv,output,dpi=600):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    df=pd.read_csv(prediction_csv);p_cols=sorted([c for c in df.columns if c.startswith('p_')])
    if not p_cols or 'y' not in df:raise ValueError('Prediction CSV needs y,p_0,p_1,...')
    labels=list(range(len(p_cols)));cm=confusion_matrix(df.y,df[p_cols].values.argmax(1),labels=labels)
    fig,ax=plt.subplots(figsize=(5,4.5));im=ax.imshow(cm,cmap='Blues')
    ax.set_xticks(labels,labels);ax.set_yticks(labels,labels)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Count');fig.tight_layout()
    Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=dpi);plt.close(fig)


def quality_score_plot(augmentation_dir,output,dpi=600):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(augmentation_dir);parts=[]
    for fp in root.glob('candidate_scores_y*_v*.csv'):
        frame=pd.read_csv(fp)
        if len(frame):parts.append(frame)
    if not parts:raise ValueError('No quality-gate candidate-score CSV found')
    df=pd.concat(parts,ignore_index=True)
    fig,ax=plt.subplots(figsize=(6,4.5))
    for name in ('class_confidence','realism','diversity','quality'):
        ax.hist(df[name].dropna().to_numpy(),bins=20,alpha=.4,label=name)
    ax.set_xlabel('Quality component score');ax.set_ylabel('Number of candidates')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=dpi);plt.close(fig)
