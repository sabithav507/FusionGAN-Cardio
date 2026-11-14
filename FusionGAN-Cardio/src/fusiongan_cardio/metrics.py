"""Numerical evaluation for generative, anatomical and classification metrics."""
from __future__ import annotations
import numpy as np
from scipy.linalg import svdvals
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,precision_score,
                             recall_score,roc_auc_score,confusion_matrix)
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn.functional as F

def frechet_distance(real,synthetic):
    """Fréchet distance computed in the supplied independent embedding space.
    Only label as Med-FID if supplied encoder was independently trained on medical data.
    """
    x=np.asarray(real,dtype=np.float64);y=np.asarray(synthetic,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or len(x)<2 or len(y)<2 or x.shape[1]!=y.shape[1]:
        raise ValueError('FID requires >=2 samples in each BxD embedding array')
    ma,mb=x.mean(0),y.mean(0)
    a=(x-ma)/np.sqrt(len(x)-1);b=(y-mb)/np.sqrt(len(y)-1)
    # For Cx=A^T A and Cy=B^T B, trace(sqrt(Cx Cy)) equals
    # nuclear_norm(A B^T). This formulation handles rank-deficient medical
    # feature covariances gracefully, and scales with sample count rather than
    # the 2048-dimensional ResNet-50 feature space when n<2048.
    covroot_trace=float(svdvals(a@b.T,check_finite=True).sum())
    return float(max(0.0,np.sum((ma-mb)**2)+np.sum(a*a)+np.sum(b*b)-2*covroot_trace))

def kernel_inception_distance(real,synthetic,subsets=10,max_subset=1000,seed=42):
    x=np.asarray(real,dtype=np.float64);y=np.asarray(synthetic,dtype=np.float64)
    if len(x)<2 or len(y)<2:raise ValueError('KID needs >=2 features per set')
    rng=np.random.default_rng(seed);scores=[];d=x.shape[1]
    for _ in range(subsets):
        a=x[rng.choice(len(x),min(len(x),max_subset),replace=False)]
        b=y[rng.choice(len(y),min(len(y),max_subset),replace=False)]
        k_aa=((a@a.T/d)+1)**3;k_bb=((b@b.T/d)+1)**3;k_ab=((a@b.T/d)+1)**3
        m,n=len(a),len(b)
        scores.append((k_aa.sum()-np.trace(k_aa))/(m*(m-1))+(k_bb.sum()-np.trace(k_bb))/(n*(n-1))-2*k_ab.mean())
    return float(np.mean(scores)),float(np.std(scores,ddof=1)) if len(scores)>1 else 0.0

def generative_precision_recall(real,synthetic,k=3):
    """k-NN improved precision/recall support approximation (Kynkäänniemi et al.)."""
    x=np.asarray(real,dtype=np.float32);y=np.asarray(synthetic,dtype=np.float32)
    if len(x)<k+1 or len(y)<k+1:raise ValueError(f'Precision/recall require >k={k} samples per set')
    dx=euclidean_distances(x,x);dy=euclidean_distances(y,y)
    np.fill_diagonal(dx,np.inf);np.fill_diagonal(dy,np.inf)
    rx=np.partition(dx,k-1,axis=1)[:,k-1];ry=np.partition(dy,k-1,axis=1)[:,k-1]
    dxy=euclidean_distances(x,y)
    precision=np.any(dxy<=rx[:,None],axis=0).mean()
    recall=np.any(dxy<=ry[None,:],axis=1).mean()
    return float(precision),float(recall)

def classify_metrics(y_true,prob,labels=None):
    y=np.asarray(y_true,dtype=int);p=np.asarray(prob,dtype=float)
    if p.ndim!=2 or len(y)!=len(p):raise ValueError('Prediction shape mismatch')
    n=p.shape[1];pred=p.argmax(axis=1);labels=list(range(n)) if labels is None else labels
    result={'samples':len(y),'accuracy':float(accuracy_score(y,pred)),
       'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
       'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
       'macro_precision':float(precision_score(y,pred,labels=labels,average='macro',zero_division=0)),
       'macro_recall':float(recall_score(y,pred,labels=labels,average='macro',zero_division=0)),
       'per_class_recall':recall_score(y,pred,labels=labels,average=None,zero_division=0).tolist(),
       'per_class_precision':precision_score(y,pred,labels=labels,average=None,zero_division=0).tolist(),
       'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}
    try:
        result['auroc']=float(roc_auc_score(y,p[:,1]) if n==2 else roc_auc_score(y,p,labels=labels,multi_class='ovr',average='macro'))
    except ValueError:result['auroc']=None
    if n==2:
        cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
        # Class 0 = reduced EF. Define positive reduced explicitly rather than class-1 default.
        result['reduced_ef_sensitivity']=float(tn/(tn+fp)) if tn+fp else None
        result['preserved_ef_specificity']=float(tp/(tp+fn)) if tp+fn else None
    else:result['minority_class_recall']=float(result['per_class_recall'][-1])
    return result

def dice(mask1,mask2):
    a=np.asarray(mask1).astype(bool);b=np.asarray(mask2).astype(bool)
    if not a.any() and not b.any():return 1.0
    return float(2*np.logical_and(a,b).sum()/(a.sum()+b.sum()+1e-10))

def hd95(mask1,mask2,spacing=(1,1)):
    a=np.asarray(mask1).astype(bool);b=np.asarray(mask2).astype(bool)
    if not a.any() and not b.any():return 0.0
    if not a.any() or not b.any():return None
    a_edge=a ^ binary_erosion(a);b_edge=b ^ binary_erosion(b)
    a_dist=distance_transform_edt(~a_edge,sampling=spacing)
    b_dist=distance_transform_edt(~b_edge,sampling=spacing)
    distances=np.r_[a_dist[b_edge],b_dist[a_edge]]
    return float(np.percentile(distances,95))

@torch.no_grad()
def get_embeddings(model,loader,device,limit=None):
    model.eval();embeds=[];total=0
    for batch in loader:
        x=batch['image'].to(device)
        if limit:x=x[:max(0,limit-total)]
        if not len(x):break
        embeds.append(model(x).float().cpu().numpy());total+=len(x)
        if limit and total>=limit:break
    return np.concatenate(embeds,axis=0)
