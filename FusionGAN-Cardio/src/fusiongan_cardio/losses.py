"""Adversarial, anatomy, feature-matching, mode-seeking and TV objectives."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def hinge_discriminator(real,fake):
    return F.relu(1-real['score']).mean()+F.relu(1+fake['score']).mean()

def adversarial_generator(fake):return -fake['score'].mean()

def conditional_loss(output,y,view):
    return F.cross_entropy(output['class_logits'],y)+F.cross_entropy(output['view_logits'],view)

def feature_matching(real,fake):
    """Mean absolute difference across four selected discriminator feature layers.
    Real activations are detached, synthetic activations remain differentiable.
    """
    if len(real['features'])!=len(fake['features']):raise ValueError('Feature count mismatch')
    terms=[]
    for a,b in zip(real['features'],fake['features']):
        # Equalize sample-to-sample mismatch by aligning batch-averaged statistics.
        # This does not force unrelated real/fake samples into pixelwise alignment.
        terms.append(F.l1_loss(b.mean(dim=0),a.detach().mean(dim=0)))
    return torch.stack(terms).mean()

def mode_seeking(z1,z2,x1,x2,eps=1e-8):
    dz=(z1-z2).abs().flatten(1).mean(1)
    dx=(x1-x2).abs().flatten(1).mean(1)
    return (dz/(dx+eps)).clamp(max=1e4).mean()

def total_variation(x):
    return (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()+(x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean()

def soft_dice_loss(seg_logits,target,num_classes=4):
    probs=torch.softmax(seg_logits,dim=1)
    one_hot=F.one_hot(target.long(),num_classes).permute(0,3,1,2).float()
    dims=(0,2,3)
    intersection=(probs*one_hot).sum(dims)
    denom=(probs+one_hot).sum(dims)
    return 1-((2*intersection[1:]+1)/(denom[1:]+1)).mean()

def supervised_segmentation_loss(logits,targets):
    return F.cross_entropy(logits,targets.long())+soft_dice_loss(logits,targets.long())

class ReferenceBank:
    """Real-only, same-class/view structure embeddings. Use train split only."""
    def __init__(self,features,classes,views):
        self.features=features.float().detach()
        self.classes=classes.long().detach();self.views=views.long().detach()
        if self.features.ndim!=2:raise ValueError('Bank embeddings must be BxD')
        if self.features.shape[0]!=self.classes.numel() or self.features.shape[0]!=self.views.numel():
            raise ValueError('Bank shapes mismatch')
    @classmethod
    @torch.no_grad()
    def from_loader(cls,encoder,loader,device,limit=None):
        encoder.eval();f=[];y=[];v=[];remaining=limit if limit else float('inf')
        for batch in loader:
            if remaining<=0:break
            mask=~batch['synthetic'].bool();x=batch['image'][mask];
            if len(x)==0:continue
            keep=min(len(x),int(remaining)) if remaining!=float('inf') else len(x)
            x=x[:keep].to(device)
            f.append(encoder(x).detach().float().cpu());y.append(batch['y'][mask][:keep]);v.append(batch['view'][mask][:keep]);remaining-=keep
        if not f:raise ValueError('No real samples available for the structure bank')
        return cls(torch.cat(f),torch.cat(y),torch.cat(v))
    def nearest(self,generated_features,y,view,k=1,exclude_self=False):
        """Differentiable wrt generated_features; neighbour indices are discrete."""
        bank=self.features.to(generated_features.device)
        cs=self.classes.to(generated_features.device)
        vs=self.views.to(generated_features.device)
        result=[];distances=[]
        for i in range(generated_features.shape[0]):
            group=bank[(cs==y[i]) & (vs==view[i])]
            if len(group)<1:
                # Caller must not claim same-condition anatomy if the group is absent.
                raise ValueError(f'Empty real reference group (class={int(y[i])},view={int(view[i])})')
            dist=torch.cdist(generated_features[i:i+1].float(),group.float(),p=2).squeeze(0)
            idx=dist.topk(min(k,len(group)),largest=False).indices
            result.append(group[idx[0]])
            distances.append(dist[idx].mean())
        return torch.stack(result),torch.stack(distances)
    def structure_loss(self,generated_features,y,view):
        nearest,_=self.nearest(generated_features,y,view)
        return F.l1_loss(generated_features.float(),nearest.detach().float())

@torch.no_grad()
def generate_real_bank(encoder,frame,cfg,device):
    from .data import dataloader
    real=frame[(frame['split']=='train') & (~frame['is_synthetic'])].copy()
    limit=cfg['training'].get('reference_bank_limit')
    if limit and len(real)>limit:
        import numpy as np
        rng=np.random.default_rng(cfg['project']['seed'])
        group_ids=[g.index.to_numpy() for _,g in real.groupby(['y','view'])]
        minimum=2*len(group_ids)
        if limit<minimum:raise ValueError(f'Reference bank limit {limit} too small: need >=2 examples for each of {len(group_ids)} condition groups')
        sampled=[]
        for ids in group_ids:
            sampled.extend(rng.choice(ids,size=min(2,len(ids)),replace=False).tolist())
        remaining=real.index.difference(sampled).to_numpy()
        extra=rng.choice(remaining,size=min(limit-len(sampled),len(remaining)),replace=False)
        real=real.loc[sampled+extra.tolist()].copy()
    loader=dataloader(real,cfg,batch_size=cfg['training']['batch_size'],shuffle=False)
    return ReferenceBank.from_loader(encoder,loader,device,limit=None)
