"""Train operational comparator baselines, with independent validation Fréchet selection.
Original StyleGAN2-ADA/ViTGAN repositories are NOT bundled. Their explicit
'_lite' names identify self-contained architectural approximations.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from .data import load_manifest,dataloader
from .utils import set_seed,get_device,grad_scaler,amp_context,save_checkpoint,load_checkpoint,freeze,jsonl_append
from .models.baselines import build_baseline
from .models.diffusion import ConditionalDiffusion
from .train import load_evaluator,validation_fid,_optim_step
from .losses import hinge_discriminator,adversarial_generator,conditional_loss

GAN_BASELINES={'dcgan','cgan','wgangp','stylegan2ada_lite','vitgan_lite'}
DIFFUSION_BASELINES={'ddpm','ldm'}

def _gradient_penalty(d,real,fake,y,v):
    eps=torch.rand((len(real),1,1,1),device=real.device)
    mixed=(eps*real+(1-eps)*fake.detach()).requires_grad_(True)
    logits=d(mixed,y,v)['score']
    grads=torch.autograd.grad(logits.sum(),mixed,create_graph=True,retain_graph=True)[0]
    return ((grads.reshape(len(grads),-1).norm(2,dim=1)-1)**2).mean()

def _ada_augmentation(x,p):
    # Restricted differentiable augmentation: NO horizontal flips, no laterality inversion.
    if p<=0:return x
    mask=(torch.rand(len(x),1,1,1,device=x.device)<p).float()
    brightness=torch.empty(len(x),1,1,1,device=x.device).uniform_(-.08,.08)
    contrast=torch.empty(len(x),1,1,1,device=x.device).uniform_(.9,1.1)
    return (x+mask*((x*contrast+brightness).clamp(0,1)-x)).clamp(0,1)

def fit_baseline(cfg,manifest,name,output,evaluator_checkpoint,task='tmed',group_class=None,group_view=None):
    if name not in GAN_BASELINES|DIFFUSION_BASELINES:raise ValueError(f'Unknown baseline {name}')
    set_seed(cfg['project']['seed']);device=get_device(cfg['project']['device'])
    df=load_manifest(manifest,task);df=df[~df['is_synthetic']]
    classes=3 if task=='tmed' else 2;views=5 if task=='tmed' else 4
    validation_source='official_validation'
    if name in ('dcgan','wgangp'):
        if group_class is None or group_view is None:
            raise ValueError('Unconditional DCGAN/WGAN-GP require --group-class and --group-view; train one checkpoint per populated condition group.')
        df=df[(df['y']==int(group_class))&(df['view']==int(group_view))].copy()
        train_ids=df.index[df['split']=='train'].tolist();val_count=int((df['split']=='val').sum())
        if val_count<2:
            # Rare groups may have no real validation examples; reserve real TRAIN
            # group images for checkpoint selection, never touch original test.
            need=2-val_count
            reserve=max(need,round(len(train_ids)*.2))
            if len(train_ids)-reserve<2:
                raise ValueError('Unconditional group too small: need >=2 train and >=2 independent validation samples after a within-train holdout.')
            rng=np.random.default_rng(cfg['project']['seed'])
            heldout=rng.choice(train_ids,reserve,replace=False)
            df.loc[heldout,'split']='val'
            validation_source='within_training_group_holdout_plus_official_validation'
    train=dataloader(df,cfg,'train',shuffle=True)
    validation=dataloader(df,cfg,'val',shuffle=False,batch_size=cfg['benchmark']['eval_batch_size'])
    evaluator,_=load_evaluator(evaluator_checkpoint,device)
    p=cfg['training'];zdim=cfg['model']['latent_dim'];path=Path(output)
    max_epochs=p['epochs'];best=float('inf');stale=0
    if name in GAN_BASELINES:
        gen,disc=build_baseline(name,cfg,classes,views);gen.to(device);disc.to(device)
        gopt=torch.optim.Adam(gen.parameters(),lr=p['lr_g'],betas=(p['beta1'],p['beta2']))
        dopt=torch.optim.Adam(disc.parameters(),lr=p['lr_d'],betas=(p['beta1'],p['beta2']))
        gs=grad_scaler(device,cfg['project']['amp']);ds=grad_scaler(device,cfg['project']['amp'])
        ada_p=0.0
    else:
        gen=ConditionalDiffusion(cfg['data']['image_size'],classes,views,cfg['model']['discriminator_base'],latent=name=='ldm').to(device)
        if name=='ldm':
            aeopt=torch.optim.AdamW(gen.vae.parameters(),lr=p['lr_g'])
            scaler=grad_scaler(device,cfg['project']['amp'])
            for ep in range(1,max(1,int(cfg['auxiliary']['epochs']))+1):
                gen.vae.train();recon=[]
                for b in train:
                    with amp_context(device,cfg['project']['amp']):
                        _,l=gen.vae(b['image'].to(device))
                    _optim_step(l,aeopt,scaler,gen.vae);recon.append(float(l.item()))
                jsonl_append(path.with_suffix('.vae.jsonl'),{'epoch':ep,'reconstruction_loss':float(np.mean(recon))})
            freeze(gen.vae)
        opt=torch.optim.AdamW(gen.noise_model.parameters(),lr=p['lr_g'])
        gs=grad_scaler(device,cfg['project']['amp'])
    for ep in range(1,max_epochs+1):
        gen.train()
        if name in GAN_BASELINES:disc.train()
        records=[]
        for b in train:
            x=b['image'].to(device);y=b['y'].to(device);v=b['view'].to(device)
            if name in GAN_BASELINES:
                z=torch.randn(len(x),zdim,device=device)
                for pp in disc.parameters():pp.requires_grad_(True)
                with amp_context(device,cfg['project']['amp']):
                    fake=gen(z,y,v).detach()
                    if name=='stylegan2ada_lite':
                        real_d=_ada_augmentation(x,ada_p);fake_d=_ada_augmentation(fake,ada_p)
                    else:real_d=x;fake_d=fake
                    real_out=disc(real_d,y,v);fake_out=disc(fake_d,y,v)
                    if name=='wgangp':
                        # GP computation uses fp32 and is not controlled by autocast.
                        with torch.autocast(device_type='cuda',enabled=False) if device.type=='cuda' else torch.enable_grad():
                            gp=_gradient_penalty(disc,x.float(),fake.float(),y,v)
                        dloss=fake_out['score'].mean()-real_out['score'].mean()+10*gp
                    else:
                        dloss=hinge_discriminator(real_out,fake_out)
                        if name not in ('dcgan',):dloss=dloss+conditional_loss(real_out,y,v)
                _optim_step(dloss,dopt,ds,disc,p.get('grad_clip',0))
                if name=='stylegan2ada_lite':
                    # Low-bandwidth ADA heuristic based on real-score sign, updated per batch.
                    accuracy=(real_out['score'].detach()>0).float().mean().item()
                    ada_p=float(np.clip(ada_p+.002*(accuracy-.6),0,.8))
                for pp in disc.parameters():pp.requires_grad_(False)
                disc.eval()
                with amp_context(device,cfg['project']['amp']):
                    fake=gen(torch.randn(len(x),zdim,device=device),y,v)
                    fake_out=disc(_ada_augmentation(fake,ada_p) if name=='stylegan2ada_lite' else fake,y,v)
                    gloss=-fake_out['score'].mean()
                    if name not in ('dcgan','wgangp'):gloss=gloss+conditional_loss(fake_out,y,v)
                _optim_step(gloss,gopt,gs,gen,p.get('grad_clip',0))
                records.append(float(gloss.item()));disc.train()
            else:
                with amp_context(device,cfg['project']['amp']):loss=gen.loss(x,y,v)
                _optim_step(loss,opt,gs,gen.noise_model,p.get('grad_clip',0))
                records.append(float(loss.item()))
        row={'epoch':ep,'train_generator_loss':float(np.mean(records))}
        if ep%p['val_every']==0 or ep==1 or ep==max_epochs:
            if name in DIFFUSION_BASELINES:
                # Fréchet evaluation remains independent of the diffusion loss.
                class _Sampler:
                    def __init__(self,m):self.m=m
                    def eval(self):self.m.eval()
                    def __call__(self,z,y,v):return self.m.sample(len(z),y,v,steps=20)
                eval_gen=_Sampler(gen)
            else:eval_gen=gen
            score=validation_fid(eval_gen,evaluator,validation,device,zdim)
            row['validation_med_fid']=score
            if score<best-1e-6:
                best=score;stale=0
                saved={'model':gen.state_dict(),'name':name,'task':task,'num_classes':classes,'num_views':views,
                     'epoch':ep,'best_fid':best,'model_config':cfg['model'],'image_size':cfg['data']['image_size'],
                     'group_class':group_class,'group_view':group_view,'implementation':'self-contained reference baseline','validation_source':validation_source}
                save_checkpoint(path,**saved)
            else:stale+=1
        jsonl_append(path.with_suffix('.jsonl'),row)
        if stale>=p['patience']:break
    return {'checkpoint':str(path),'best_val_fid':best,'epochs':ep,'baseline':name}

def load_baseline(path,device='cpu'):
    ck=load_checkpoint(path,device)
    cfg={'model':ck['model_config'],'data':{'image_size':ck['image_size']}}
    if ck['name'] in GAN_BASELINES:
        gen,_=build_baseline(ck['name'],cfg,ck['num_classes'],ck['num_views'])
    else:
        gen=ConditionalDiffusion(ck['image_size'],ck['num_classes'],ck['num_views'],ck['model_config']['discriminator_base'],latent=ck['name']=='ldm')
    gen.load_state_dict(ck['model']);return freeze(gen.to(device)),ck
