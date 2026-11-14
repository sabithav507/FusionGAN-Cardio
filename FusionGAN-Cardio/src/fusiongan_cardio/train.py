"""Complete training loops for auxiliary networks, FusionGAN, teacher and classifiers."""
from __future__ import annotations
import copy, math, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from .config import dump_config
from .data import load_manifest,dataloader, EchoImageDataset
from .utils import get_device, amp_context, grad_scaler, save_checkpoint, load_checkpoint, freeze, set_seed,jsonl_append
from .losses import (hinge_discriminator,adversarial_generator,feature_matching,conditional_loss,
                     mode_seeking,total_variation,supervised_segmentation_loss,generate_real_bank)
from .models import (make_gan,create_structure,UNetProbe,create_classifier)
from .models.evaluation import SimCLREncoder,contrastive_loss,augment_ultrasound
from .metrics import classify_metrics,frechet_distance

def _optim_step(loss,optimizer,scaler,model=None,grad_clip=0):
    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {loss.item()}')
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip and model is not None:
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),float(grad_clip))
    scaler.step(optimizer);scaler.update()

def _classification_eval(model,loader,device):
    model.eval();pred=[];targets=[];paths=[];studies=[];subjects=[]
    with torch.no_grad():
        for b in loader:
            p=F.softmax(model(b['image'].to(device)),dim=1).float().cpu().numpy()
            pred.append(p);targets.extend(b['y'].tolist())
            paths.extend(b['path']);studies.extend(b['study_id']);subjects.extend(b['subject_id'])
    if not pred:raise ValueError('Empty classification evaluation set')
    return np.asarray(targets),np.concatenate(pred),paths,studies,subjects

def fit_classifier(cfg,manifest,output,task='tmed',architecture=None,teacher=False,seed=None):
    """Teacher DenseNet-121 (real only) OR fixed downstream EfficientNet-B0."""
    seed=cfg['project']['seed'] if seed is None else int(seed);set_seed(seed)
    device=get_device(cfg['project']['device']);df=load_manifest(manifest,task)
    if teacher:df=df[~df['is_synthetic']].copy()
    classes=3 if task=='tmed' else 2
    arch=architecture or ('densenet121' if teacher else cfg['classifier']['architecture'])
    model=create_classifier(arch,classes,pretrained=cfg['model'].get('pretrained',False)).to(device)
    part=cfg['auxiliary'] if teacher else cfg['classifier']
    batch=part.get('batch_size',32)
    tr=dataloader(df,cfg,'train',augment=not teacher,batch_size=batch,balanced=not teacher and df['is_synthetic'].any())
    va=dataloader(df,cfg,'val',batch_size=batch,shuffle=False)
    opt=torch.optim.AdamW(model.parameters(),lr=part['lr'],weight_decay=part.get('weight_decay',1e-4))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,part['epochs']))
    scaler=grad_scaler(device,cfg['project']['amp'])
    best=-float('inf');stale=0;path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    log=path.with_suffix('.jsonl');log.unlink(missing_ok=True)
    for epoch in range(1,part['epochs']+1):
        model.train();total=0.;count=0
        for b in tr:
            x=b['image'].to(device);y=b['y'].to(device)
            with amp_context(device,cfg['project']['amp']):
                logits=model(x);loss=F.cross_entropy(logits,y)
            _optim_step(loss,opt,scaler,model)
            total+=float(loss.item())*len(x);count+=len(x)
        scheduler.step()
        y,p,*_= _classification_eval(model,va,device)
        metrics=classify_metrics(y,p)
        score=metrics['macro_f1']
        jsonl_append(log,{'epoch':epoch,'train_loss':total/max(1,count),'val_macro_f1':score,'val_bal_acc':metrics['balanced_accuracy']})
        if score>best+1e-6:
            best=score;stale=0
            save_checkpoint(path,model=model.state_dict(),architecture=arch,num_classes=classes,task=task,epoch=epoch,
                            validation_metrics=metrics,seed=seed,teacher=teacher,image_size=cfg['data']['image_size'])
        else:stale+=1
        if stale>=part.get('patience',10):break
    return {'checkpoint':str(path),'best_validation_macro_f1':best,'epochs':epoch,'teacher':teacher}

def load_classifier(path,device='cpu'):
    state=load_checkpoint(path,device)
    model=create_classifier(state['architecture'],state['num_classes'],pretrained=False)
    model.load_state_dict(state['model']);model.to(device);freeze(model)
    return model,state

@torch.no_grad()
def evaluate_classifier(cfg,manifest,checkpoint,split='test',output=None):
    import pandas as pd
    device=get_device(cfg['project']['device']);model,info=load_classifier(checkpoint,device)
    df=load_manifest(manifest,info['task'])
    dataset=dataloader(df,cfg,split,batch_size=cfg['benchmark'].get('eval_batch_size',32),shuffle=False)
    y,p,paths,studies,subjects=_classification_eval(model,dataset,device)
    report={'image_level':classify_metrics(y,p),'task':info['task'],'split':split,
            'label_description':'AS study-label image-level proxy' if info['task']=='tmed' else 'static frame proxy for reduced/preserved EF'}
    preds=pd.DataFrame({'path':paths,'y':y,'study_id':studies,'subject_id':subjects})
    for i in range(p.shape[1]):preds[f'p_{i}']=p[:,i]
    # Study/video-level aggregation uses independent subject/study as the unit.
    grouped=preds.groupby('study_id',sort=True)
    aggregated=grouped[[f'p_{i}' for i in range(p.shape[1])]].mean()
    truth=grouped.y.first()
    if (grouped.y.nunique()!=1).any():raise ValueError('Conflicting labels within a study: cannot aggregate')
    report['study_level']=classify_metrics(truth.to_numpy(),aggregated.to_numpy())
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        preds.to_csv(output/f'{split}_predictions.csv',index=False)
        aggregated.assign(y=truth).to_csv(output/f'{split}_study_predictions.csv')
        from .utils import write_json
        write_json(output/f'{split}_metrics.json',report)
    return report

def fit_structure(cfg,camus_manifest,echonet_manifest,output):
    """Independent ResNet-18 anatomy encoder: CAMUS mask supervision + EchoNet EF supervision.
    CAMUS heldout 100 subjects NEVER participate in optimization or checkpoint selection.
    """
    set_seed(cfg['project']['seed']);device=get_device(cfg['project']['device'])
    camus=load_manifest(camus_manifest,'camus');echonet=load_manifest(echonet_manifest,'echonet')
    a=dataloader(camus,cfg,'train',batch_size=cfg['auxiliary']['batch_size'],shuffle=True,masks=True)
    b=dataloader(echonet,cfg,'train',batch_size=cfg['auxiliary']['batch_size'],shuffle=True)
    encoder=create_structure(cfg).to(device)
    opt=torch.optim.AdamW(encoder.parameters(),lr=cfg['auxiliary']['lr'])
    scaler=grad_scaler(device,cfg['project']['amp']);best=float('inf');stale=0;path=Path(output)
    log=path.with_suffix('.jsonl');log.unlink(missing_ok=True)
    for ep in range(1,cfg['auxiliary']['epochs']+1):
        encoder.train();losses=[]
        from itertools import zip_longest,cycle,islice
        nsteps=max(len(a),len(b))
        ca=iter(a);eb=iter(b)
        for _ in range(nsteps):
            try:c=next(ca)
            except StopIteration:ca=iter(a);c=next(ca)
            try:e=next(eb)
            except StopIteration:eb=iter(b);e=next(eb)
            with amp_context(device,cfg['project']['amp']):
                _,seg,_=encoder(c['image'].to(device),return_heads=True)
                _,_,ef_logits=encoder(e['image'].to(device),return_heads=True)
                loss=supervised_segmentation_loss(seg,c['mask'].to(device))+F.cross_entropy(ef_logits,e['y'].to(device))
            _optim_step(loss,opt,scaler,encoder);losses.append(float(loss.item()))
        # Only EchoNet official validation is used for checkpoint selection (CAMUS 100 held-out remain untouched).
        encoder.eval();val_losses=[]
        with torch.no_grad():
            if (echonet['split']=='val').any():
                ev=dataloader(echonet,cfg,'val',batch_size=cfg['auxiliary']['batch_size'],shuffle=False)
                for batch in ev:
                    _,_,logits=encoder(batch['image'].to(device),return_heads=True)
                    val_losses.append(float(F.cross_entropy(logits,batch['y'].to(device)).item()))
        score=float(np.mean(val_losses if val_losses else losses))
        jsonl_append(log,{'epoch':ep,'training_loss':float(np.mean(losses)),'echonet_val_ef_loss':score})
        if score<best-1e-6:
            best=score;stale=0
            save_checkpoint(path,model=encoder.state_dict(),small=cfg['model']['base_channels']<16,
                            embed_dim=cfg['auxiliary']['encoder_dim'],epoch=ep,
                            objective='CAMUS mask + EchoNet train EF; model selection by EchoNet validation EF')
        else:stale+=1
        if stale>=cfg['auxiliary'].get('patience',8):break
    return {'checkpoint':str(path),'epochs':ep,'best_val_loss':best}

def load_structure(path,device='cpu'):
    from .models.support import TinyEncoder,StructureEncoder
    info=load_checkpoint(path,device)
    enc=TinyEncoder(info['embed_dim']) if info.get('small',False) else StructureEncoder(info['embed_dim'],pretrained=False)
    enc.load_state_dict(info['model']);enc.to(device);freeze(enc)
    return enc,info

def fit_probe(cfg,camus_manifest,output):
    set_seed(cfg['project']['seed']);device=get_device(cfg['project']['device'])
    df=load_manifest(camus_manifest,'camus')
    loader=dataloader(df,cfg,'train',masks=True,batch_size=cfg['auxiliary']['batch_size'],shuffle=True)
    model=UNetProbe(base=8 if cfg['data']['image_size']<64 else 24).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['auxiliary']['lr']);scaler=grad_scaler(device,cfg['project']['amp'])
    path=Path(output)
    for ep in range(1,cfg['auxiliary']['epochs']+1):
        model.train();losses=[]
        for b in loader:
            with amp_context(device,cfg['project']['amp']):
                p=model(b['image'].to(device));loss=supervised_segmentation_loss(p,b['mask'].to(device))
            _optim_step(loss,opt,scaler,model);losses.append(float(loss.item()))
        jsonl_append(path.with_suffix('.jsonl'),{'epoch':ep,'train_mask_loss':float(np.mean(losses))})
    save_checkpoint(path,model=model.state_dict(),base=8 if cfg['data']['image_size']<64 else 24,
                    epoch=ep,selection='training-only, CAMUS heldout not used')
    return {'checkpoint':str(path),'epochs':ep}

def load_probe(path,device='cpu'):
    ckpt=load_checkpoint(path,device);model=UNetProbe(base=ckpt['base']).to(device)
    model.load_state_dict(ckpt['model']);freeze(model)
    return model,ckpt

def fit_evaluator(cfg,manifest,output,epochs=None,unlabeled_manifest=None):
    """Self-supervised 2-view SimCLR; only real training split is read.
    A labeled-only training subset is acceptable engineering fallback but must
    be disclosed; it is not the manuscript's exact unlabeled pretraining cohort.
    """
    set_seed(cfg['project']['seed']);device=get_device(cfg['project']['device'])
    df=load_manifest(manifest,'tmed');labeled=df.copy()
    df=df[(df['split']=='train')&(~df['is_synthetic'])]
    pretraining_source='real training-only TMED-2 labeled cohort; not exact unlabelled cohort'
    if unlabeled_manifest:
        import pandas as pd
        raw=pd.read_csv(unlabeled_manifest)
        required={'path','study_id','subject_id'}
        if not required.issubset(raw):raise ValueError('Unlabeled manifest requires path,study_id,subject_id')
        if raw[['path','study_id','subject_id']].isna().any().any():raise ValueError('Missing unlabelled manifest identifiers')
        if 'split' in raw and not raw['split'].astype(str).str.lower().isin(['train','training','tr']).all():
            raise ValueError('SimCLR unlabeled manifest includes non-training partition images')
        # Reserve patient/study isolation from all labelled validation and test studies.
        forbidden=labeled[labeled['split']!='train']
        if set(raw['subject_id'].astype(str))&set(forbidden['subject_id'].astype(str)) or set(raw['study_id'].astype(str))&set(forbidden['study_id'].astype(str)):
            raise ValueError('Unlabelled pretraining cohort overlaps labelled validation or test patients/studies')
        root=Path(unlabeled_manifest).resolve().parent
        raw['path']=raw['path'].map(lambda x:str((root/str(x)).resolve()) if not Path(str(x)).is_absolute() else str(Path(str(x)).resolve()))
        if not all(Path(x).is_file() for x in raw['path']):raise FileNotFoundError('Some unlabelled images do not exist')
        if raw['path'].duplicated().any():raise ValueError('Duplicated unlabelled paths')
        raw['split']='train';raw['y']=0;raw['view']=0;raw['is_synthetic']=False
        df=raw
        pretraining_source='explicit TMED-2 unlabelled image manifest, validated disjoint from labelled val/test'
    loader=dataloader(df,cfg,batch_size=max(2,cfg['auxiliary']['batch_size']),shuffle=True)
    small=cfg['model']['base_channels']<16
    model=SimCLREncoder(small=small).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['auxiliary']['lr'])
    scaler=grad_scaler(device,cfg['project']['amp']);path=Path(output)
    for ep in range(1,(epochs or cfg['auxiliary']['epochs'])+1):
        model.train();losses=[]
        for b in loader:
            x=b['image'].to(device)
            if len(x)<2:continue
            with amp_context(device,cfg['project']['amp']):
                a=model(augment_ultrasound(x),project=True)
                c=model(augment_ultrasound(x),project=True)
                loss=contrastive_loss(a,c,cfg['auxiliary'].get('simclr_temperature',.2))
            _optim_step(loss,opt,scaler,model);losses.append(float(loss.item()))
        if not losses:raise ValueError('SimCLR training requires batch size >=2')
        jsonl_append(path.with_suffix('.jsonl'),{'epoch':ep,'train_contrastive_loss':float(np.mean(losses))})
    save_checkpoint(path,model=model.state_dict(),small=small,epoch=ep,feature_dim=32 if small else 2048,
                    feature_source=pretraining_source)
    return {'checkpoint':str(path),'epochs':ep}

def load_evaluator(path,device='cpu'):
    ckpt=load_checkpoint(path,device)
    model=SimCLREncoder(small=ckpt['small']).to(device);model.load_state_dict(ckpt['model']);freeze(model)
    return model,ckpt

@torch.no_grad()
def validation_fid(gen,evaluator,loader,device,latent_dim):
    gen.eval();evaluator.eval();real=[];fake=[]
    for b in loader:
        x=b['image'].to(device);y=b['y'].to(device);v=b['view'].to(device)
        z=torch.randn(len(x),latent_dim,device=device)
        with torch.no_grad():syn=gen(z,y,v)
        real.append(evaluator(x).float().cpu().numpy())
        fake.append(evaluator(syn).float().cpu().numpy())
    return frechet_distance(np.concatenate(real),np.concatenate(fake))

def fit_gan(cfg,manifest,structure_checkpoint,evaluator_checkpoint,output,resume=None,task='tmed'):
    """Alternating adversarial optimization with exact train-only anatomical bank."""
    set_seed(cfg['project']['seed']);device=get_device(cfg['project']['device'])
    df=load_manifest(manifest,task);df=df[~df['is_synthetic']].copy()  # GAN is trained only on real images
    classes=3 if task=='tmed' else 2;views=5 if task=='tmed' else 4
    tr=dataloader(df[df['split']=='train'],cfg,'train',shuffle=True)
    va=dataloader(df,cfg,'val',batch_size=cfg['benchmark']['eval_batch_size'],shuffle=False)
    structure,_=load_structure(structure_checkpoint,device)
    evaluator,_=load_evaluator(evaluator_checkpoint,device)
    bank=generate_real_bank(structure,df,cfg,device)
    gen,disc=make_gan(cfg,classes,views);gen=gen.to(device);disc=disc.to(device)
    p=cfg['training'];w=p['weights'];zdim=cfg['model']['latent_dim']
    go=torch.optim.Adam(gen.parameters(),lr=p['lr_g'],betas=(p['beta1'],p['beta2']))
    do=torch.optim.Adam(disc.parameters(),lr=p['lr_d'],betas=(p['beta1'],p['beta2']))
    gs=grad_scaler(device,cfg['project']['amp']);ds=grad_scaler(device,cfg['project']['amp'])
    begin=1;best=float('inf');stale=0
    if resume:
        ckpt=load_checkpoint(resume,device)
        gen.load_state_dict(ckpt['generator']);disc.load_state_dict(ckpt['discriminator'])
        go.load_state_dict(ckpt['g_optimizer']);do.load_state_dict(ckpt['d_optimizer'])
        begin=ckpt['epoch']+1;best=ckpt.get('best_fid',best);stale=ckpt.get('stale',0)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    log=path.with_suffix('.jsonl')
    if not resume:log.unlink(missing_ok=True)
    for ep in range(begin,p['epochs']+1):
        gen.train();disc.train();entries=[]
        for b in tr:
            x=b['image'].to(device);y=b['y'].to(device);v=b['view'].to(device);n=len(x)
            z=torch.randn(n,zdim,device=device)
            # D update: train auxiliary heads on *real* class/view labels.
            for par in disc.parameters():par.requires_grad_(True)
            with amp_context(device,cfg['project']['amp']):
                xfake=gen(z,y,v).detach()
                real_out=disc(x,y,v);fake_out=disc(xfake,y,v)
                d_hinge=hinge_discriminator(real_out,fake_out)
                d_aux=conditional_loss(real_out,y,v)
                d_loss=d_hinge+d_aux
            _optim_step(d_loss,do,ds,disc,p.get('grad_clip',0))
            # G update: discriminator frozen, but gradients through D and E_s to synthetic x remain.
            disc.eval()
            for par in disc.parameters():par.requires_grad_(False)
            with amp_context(device,cfg['project']['amp']):
                fake=gen(z,y,v)
                predicted=disc(fake,y,v)
                with torch.no_grad():reference=disc(x,y,v)
                terms={}
                terms['adv']=adversarial_generator(predicted)
                terms['fm']=feature_matching(reference,predicted)
                terms['cv']=conditional_loss(predicted,y,v)
                if cfg['model'].get('ablation',{}).get('structure_loss',True):
                    encoded=structure(fake)  # gradient to fake; encoder parameters frozen
                    terms['structure']=bank.structure_loss(encoded,y,v)
                else:terms['structure']=fake.sum()*0
                z2=torch.randn_like(z);fake2=gen(z2,y,v)
                terms['mode']=mode_seeking(z,z2,fake,fake2)
                terms['tv']=total_variation(fake)
                g_loss=sum(float(w[k])*t for k,t in terms.items())
            _optim_step(g_loss,go,gs,gen,p.get('grad_clip',0))
            entries.append({'d':float(d_loss.item()),'g':float(g_loss.item()),
                            **{k:float(val.item()) for k,val in terms.items()}})
            disc.train()
        record={'epoch':ep,**{k:float(np.mean([b[k] for b in entries])) for k in entries[0]}}
        evaluate=(ep%p['val_every']==0 or ep==1 or ep==p['epochs'])
        if evaluate:
            score=validation_fid(gen,evaluator,va,device,zdim)
            record['validation_med_fid']=score
            if score<best-1e-6:
                best=score;stale=0
                save_checkpoint(path,generator=gen.state_dict(),discriminator=disc.state_dict(),
                                epoch=ep,best_fid=best,task=task,num_classes=classes,num_views=views,
                                model_config=cfg['model'],image_size=cfg['data']['image_size'],
                                selection='validation medical-trained SimCLR Fréchet',
                                evaluator_checkpoint=str(evaluator_checkpoint),structure_checkpoint=str(structure_checkpoint))
            else:stale+=1
        jsonl_append(log,record)
        save_checkpoint(path.with_name('last_'+path.name),generator=gen.state_dict(),discriminator=disc.state_dict(),
                        g_optimizer=go.state_dict(),d_optimizer=do.state_dict(),epoch=ep,best_fid=best,stale=stale,
                        task=task,num_classes=classes,num_views=views,model_config=cfg['model'],image_size=cfg['data']['image_size'])
        if stale>=p['patience']:break
    return {'checkpoint':str(path),'epochs':ep,'best_validation_med_fid':best}

def load_gan(path,device='cpu'):
    """Reconstruct the exact saved architecture without accessing training data."""
    from .models.core import FusionGenerator,MultiScaleDiscriminator
    state=load_checkpoint(path,device);m=state['model_config'];a=m.get('ablation',{})
    g=FusionGenerator(state['num_classes'],state['num_views'],state['image_size'],m['latent_dim'],
      m['condition_embed_dim'],m['base_channels'],m['fused_channels'],m.get('attention_reduction',8),
      a.get('adaptive_fusion',True),a.get('dual_attention',True))
    d=MultiScaleDiscriminator(state['num_classes'],state['num_views'],m['discriminator_base'],state['image_size'],
      a.get('multi_scale',True))
    g.load_state_dict(state['generator']);d.load_state_dict(state['discriminator'])
    return freeze(g.to(device)),freeze(d.to(device)),state
