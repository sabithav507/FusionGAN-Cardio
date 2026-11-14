"""Reproducible disease-view quality gating, deficit-driven balancing and provenance."""
from __future__ import annotations
from pathlib import Path
import math, warnings, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from .data import load_manifest,EchoImageDataset,dataloader,audit_splits
from .utils import get_device,set_seed,save_checkpoint,load_checkpoint,write_json,freeze,grad_scaler,amp_context
from .train import load_classifier,load_structure,load_gan,_optim_step
from .baseline_train import load_baseline
from .losses import generate_real_bank,ReferenceBank
from .models.support import create_classifier

class QualityGate:
    def __init__(self,teacher,structure,bank,config,realism_discriminator=None,realism_critic=None):
        self.teacher=freeze(teacher);self.structure=freeze(structure);self.bank=bank
        self.d=freeze(realism_discriminator) if realism_discriminator is not None else None
        self.critic=freeze(realism_critic) if realism_critic is not None else None
        self.cfg=config
        if self.d is None and self.critic is None:raise ValueError('A discriminator or independent realism critic is required')
        self.stats=self._compute_real_statistics()
    def _compute_real_statistics(self):
        """Leave-one-out k-NN real within-condition distance mean and SD."""
        bank=self.bank;f=bank.features.cpu().float();y=bank.classes;v=bank.views
        stats={};k=self.cfg['quality']['k_neighbors']
        for key in sorted(set(zip(y.tolist(),v.tolist()))):
            mask=(y==key[0])&(v==key[1]);emb=f[mask]
            if len(emb)<2:
                stats[key]=None;continue
            ds=torch.cdist(emb,emb).float();ds.fill_diagonal_(float('inf'))
            neigh=ds.topk(min(k,len(emb)-1),dim=1,largest=False).values.mean(dim=1)
            mu=float(neigh.mean());std=float(neigh.std(unbiased=False))
            stats[key]={'mu':mu,'std':max(std,.10*max(mu,1e-4),1e-4),'n':len(emb)}
        return stats
    @torch.no_grad()
    def score(self,images,y,views):
        """Return S_c,S_r,S_d,Q and Boolean gate output for each candidate."""
        x=images
        p=F.softmax(self.teacher(x),dim=1)
        confidence=p.gather(1,y[:,None]).squeeze(1)
        if self.critic is not None:
            realism=F.softmax(self.critic(x),dim=1)[:,1]
        else:
            realism=torch.sigmoid(self.d(x,y,views)['global_score'])
        embeddings=self.structure(x).float()
        diversity=[]
        bank=self.bank
        for i in range(len(x)):
            key=int(y[i]),int(views[i]);stats=self.stats.get(key)
            if stats is None:
                diversity.append(0.0);continue
            f=bank.features[(bank.classes==key[0])&(bank.views==key[1])].to(x.device).float()
            dist=torch.cdist(embeddings[i:i+1],f).flatten()
            avg=dist.topk(min(len(f),self.cfg['quality']['k_neighbors']),largest=False).values.mean()
            # Bell-shaped: duplicates and out-of-manifold outliers receive low scores.
            z=(avg-stats['mu'])/stats['std']
            diversity.append(float(torch.exp(-.5*z*z).clamp(0,1)))
        dscore=torch.tensor(diversity,device=x.device)
        q=self.cfg['quality']
        quality=q['class_weight']*confidence+q['realism_weight']*realism+q['diversity_weight']*dscore
        accepted=(confidence>=q['class_threshold'])&(quality>=q['combined_threshold'])
        return {'class_confidence':confidence,'realism':realism,'diversity':dscore,'quality':quality,'accepted':accepted}

def balancing_deficits(df,ratio=.9):
    """Deficit for only observed real training condition groups; never invent an absent class/view."""
    train=df[(df['split']=='train') & (~df['is_synthetic'])]
    counts=train.groupby(['y','view']).size().to_dict()
    if not counts:raise ValueError('No real training images')
    target=math.ceil(max(counts.values())*ratio)
    deficits={key:max(0,target-n) for key,n in counts.items()}
    return counts,target,deficits

def fit_realism_critic(cfg,real_manifest,candidate_manifests,output,task='tmed'):
    """ResNet-18 real-vs-synthetic common critic, trained on train data only.
    For fair eight-generator comparisons, provide pools from ALL eight generators.
    """
    device=get_device(cfg['project']['device']);set_seed(cfg['project']['seed'])
    df=load_manifest(real_manifest,task);real=df[(df['split']=='train') & (~df['is_synthetic'])].copy()
    if not candidate_manifests:raise ValueError('At least one generated candidate manifest required')
    fake=[]
    for path in candidate_manifests:
        cand=pd.read_csv(path)
        if 'path' not in cand:raise ValueError(f'Candidate CSV missing path: {path}')
        temp=cand.copy();temp['path']=temp['path'].map(lambda s:str(Path(s).resolve() if Path(s).is_absolute() else (Path(path).resolve().parent/str(s)).resolve()))
        temp['is_synthetic']=True;temp['split']='train';fake.append(temp)
    if not fake or any(len(f)==0 for f in fake):raise ValueError('No generated samples for common critic')
    # Equal contribution from each generator BEFORE equal real-versus-fake sampling.
    per_pool=min(len(f) for f in fake)
    fake=pd.concat([f.sample(n=per_pool,random_state=cfg['project']['seed']) for f in fake],ignore_index=True)
    n=min(len(real),len(fake));real=real.sample(n=n,random_state=cfg['project']['seed']);fake=fake.sample(n=n,random_state=cfg['project']['seed'])
    comb=pd.concat([real,fake],ignore_index=True);comb['y']=np.r_[np.ones(n,dtype=int),np.zeros(n,dtype=int)]
    from torch.utils.data import DataLoader
    loader=DataLoader(EchoImageDataset(comb,cfg['data']['image_size']),batch_size=cfg['auxiliary']['batch_size'],shuffle=True,
                      num_workers=cfg['project']['num_workers'])
    arch='tiny' if cfg['model']['base_channels']<16 else 'resnet18'
    model=create_classifier(arch,2,pretrained=cfg['model'].get('pretrained',False)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['auxiliary']['lr']);scaler=grad_scaler(device,cfg['project']['amp'])
    for ep in range(1,cfg['auxiliary']['epochs']+1):
        model.train()
        for b in loader:
            x=b['image'].to(device);y=b['y'].to(device)
            with amp_context(device,cfg['project']['amp']):loss=F.cross_entropy(model(x),y)
            _optim_step(loss,opt,scaler,model)
    save_checkpoint(output,model=model.state_dict(),architecture=arch,num_classes=2,
                    candidate_pool_count=len(candidate_manifests),selection='common training-only realism critic',
                    real_label_index=1)
    return {'checkpoint':str(output),'real_images':n,'fake_images':n,'epochs':ep,'candidate_manifests':candidate_manifests}

def load_realism_critic(path,device):
    ck=load_checkpoint(path,device);net=create_classifier(ck['architecture'],2,pretrained=False).to(device)
    net.load_state_dict(ck['model']);return freeze(net),ck

def _export_png(path,tensor):
    img=(tensor.detach().float().cpu().clamp(0,1).squeeze(0).numpy()*255).round().astype(np.uint8)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(img).save(path)

def fit_balanced_manifest(cfg,real_manifest,generator_checkpoint,teacher_checkpoint,structure_checkpoint,output_dir,
                          task='tmed',kind='fusion',critic_checkpoint=None,without_gating=False,
                          checkpoint_map=None,seed=None,sampling_steps=50):
    """Apply Algorithm 3. Output train augmented, val/test real-only CSV and audit report.
    For unconditional groups, optional checkpoint_map={'y:view':'path'}.
    """
    seed=cfg['project']['seed'] if seed is None else int(seed);set_seed(seed)
    device=get_device(cfg['project']['device']);df=load_manifest(real_manifest,task)
    if df['is_synthetic'].any():raise ValueError('Provide original real-only manifest to avoid recursive augmentation')
    teacher,teacher_ck=load_classifier(teacher_checkpoint,device)
    if not teacher_ck.get('teacher',False):warnings.warn('Provided quality-gate teacher checkpoint is not flagged as teacher')
    if teacher_ck.get('task')!=task or teacher_ck.get('num_classes')!=(3 if task=='tmed' else 2):
        raise ValueError('Quality-gate teacher task/class count must match the generator and dataset')
    structure,_=load_structure(structure_checkpoint,device)
    if kind=='fusion':
        gen,disc,gen_info=load_gan(generator_checkpoint,device)
    else:
        gen,gen_info=load_baseline(generator_checkpoint,device);disc=None
    if gen_info['task']!=task:raise ValueError('Generator task differs from manifest')
    if critic_checkpoint:critic,_=load_realism_critic(critic_checkpoint,device)
    else:
        critic=None
        if kind!='fusion':raise ValueError('Baseline comparisons MUST supply the common --critic-checkpoint')
    bank=generate_real_bank(structure,df,cfg,device)
    gate=QualityGate(teacher,structure,bank,cfg,disc,critic)
    counts,target,deficits=balancing_deficits(df,cfg['data']['balance_ratio'])
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    imgdir=root/'synthetic_images';imgdir.mkdir(parents=True,exist_ok=True)
    manifest_records=[];reports=[]
    qcfg=cfg['quality'];bs=int(qcfg['generation_batch']);max_attempts=int(qcfg['max_attempts_per_group'])
    rng=np.random.default_rng(seed)
    for (cls,view),needed in sorted(deficits.items()):
        if needed<1:continue
        if checkpoint_map is not None:
            key=f'{cls}:{view}'
            if key not in checkpoint_map:raise ValueError(f'Missing baseline checkpoint_map entry {key}')
            group_gen,group_info=load_baseline(checkpoint_map[key],device)
        else:
            group_gen=gen
            if gen_info.get('group_class') is not None and (gen_info['group_class']!=cls or gen_info['group_view']!=view):
                raise ValueError('Group-specific unconditional checkpoint cannot synthesize other groups; supply checkpoint_map')
        attempts=0;accepted=[];raw=[];rejected=0;record_candidates=[]
        # No-gating ablation draws random selected images from the entire raw pool.
        limit=max(max_attempts,needed) if without_gating else max_attempts
        while (len(accepted)<needed if not without_gating else len(raw)<min(limit,max(needed*2,needed+1))) and attempts<limit:
            k=min(bs,limit-attempts)
            yy=torch.full((k,),int(cls),device=device,dtype=torch.long)
            vv=torch.full((k,),int(view),device=device,dtype=torch.long)
            with torch.no_grad():
                if hasattr(group_gen,'sample'):
                    x=group_gen.sample(k,yy,vv,steps=sampling_steps)
                else:
                    latent=torch.randn(k,gen_info['model_config']['latent_dim'],device=device)
                    x=group_gen(latent,yy,vv)
                if without_gating:
                    scores=None
                else:
                    scores=gate.score(x,yy,vv)
            for i in range(k):
                attempt_id=attempts+i
                if without_gating:
                    path=imgdir/f'raw_s{seed}_y{cls}_v{view}_{attempt_id:06d}.png'
                    _export_png(path,x[i]);raw.append((path,attempt_id))
                    continue
                passed=bool(scores['accepted'][i]);entry={name:float(scores[name][i]) for name in ('class_confidence','realism','diversity','quality')}
                entry.update({'y':cls,'view':view,'attempt':attempt_id,'accepted':passed})
                record_candidates.append(entry)
                if passed and len(accepted)<needed:
                    path=imgdir/f's{seed}_y{cls}_v{view}_{attempt_id:06d}.png'
                    _export_png(path,x[i]);accepted.append((path,entry))
                else:rejected+=1
            attempts+=k
        if without_gating:
            choice=rng.choice(len(raw),min(needed,len(raw)),replace=False) if raw else []
            selected={int(index) for index in choice}
            for j,(path,attempt_id) in enumerate(raw):
                if j in selected:
                    accepted.append((path,{'class_confidence':None,'realism':None,'diversity':None,'quality':None,'attempt':attempt_id}))
                else:path.unlink(missing_ok=True)
        for j,(path,sc) in enumerate(accepted):
            subj=f'SYN_{seed}_{cls}_{view}_{j:06d}'
            manifest_records.append({'path':str(path),'split':'train','subject_id':subj,'study_id':subj,
                                     'y':cls,'view':view,'is_synthetic':True,'generator':kind,'seed':seed,
                                     **{key:sc.get(key) for key in ('class_confidence','realism','diversity','quality')}})
        if record_candidates:
            pd.DataFrame(record_candidates).to_csv(root/f'candidate_scores_y{cls}_v{view}.csv',index=False)
        reports.append({'class':int(cls),'view':int(view),'real_count':int(counts[(cls,view)]),
                        'target':target,'quota':needed,'accepted':len(accepted),'attempted':attempts,
                        'shortfall':needed-len(accepted),'acceptance_rate':len(accepted)/max(1,attempts),
                        'gate_disabled':without_gating})
        if needed>len(accepted) and not qcfg.get('allow_shortfall',True):
            raise RuntimeError(f'Quality gate deficit group {cls}:{view}: {len(accepted)}/{needed} accepted')
    original=df.copy();synthetic=pd.DataFrame(manifest_records)
    merged=pd.concat([original,synthetic],ignore_index=True,sort=False) if manifest_records else original
    audit=audit_splits(merged)
    out=root/'balanced_manifest.csv';merged.to_csv(out,index=False)
    result={'manifest':str(out),'source_manifest':str(real_manifest),'generator_checkpoint':str(generator_checkpoint),
            'seed':seed,'task':task,'quality_gate_enabled':not without_gating,'groups':reports,'audit':audit,
            'n_synthetic':len(synthetic),'n_real':len(original),'n_train':int((merged['split']=='train').sum()),
            'n_val':int((merged['split']=='val').sum()),'n_test':int((merged['split']=='test').sum())}
    write_json(root/'augmentation_report.json',result)
    return result

@torch.no_grad()
def sample_candidate_pool(cfg,real_manifest,generator_checkpoint,output_dir,task='tmed',kind='fusion',
                          samples_per_group=100,checkpoint_map=None,seed=None,sampling_steps=50):
    """Export ungated condition-stratified samples for training the SHARED critic.
    These raw candidates are NEVER added to downstream training. Use candidate pools
    from every comparator (including FusionGAN) to fit an independent critic.
    """
    seed=cfg['project']['seed'] if seed is None else int(seed);set_seed(seed)
    device=get_device(cfg['project']['device']);df=load_manifest(real_manifest,task)
    if df['is_synthetic'].any():raise ValueError('Only original real manifest allowed')
    if kind=='fusion':gen,_,info=load_gan(generator_checkpoint,device)
    else:gen,info=load_baseline(generator_checkpoint,device)
    if info['task']!=task:raise ValueError('Generator task != dataset task')
    groups,target,_=balancing_deficits(df,cfg['data']['balance_ratio'])
    output_dir=Path(output_dir).resolve();images=output_dir/'raw_images';images.mkdir(parents=True,exist_ok=True)
    bs=int(cfg['quality']['generation_batch']);records=[]
    for (cl,vi),count in sorted(groups.items()):
        if checkpoint_map is not None:
            key=f'{cl}:{vi}'
            if key not in checkpoint_map:raise ValueError(f'Missing candidate group checkpoint {key}')
            model,group_info=load_baseline(checkpoint_map[key],device)
        else:
            model,group_info=gen,info
            if group_info.get('group_class') is not None and (group_info['group_class'],group_info['group_view'])!=(cl,vi):
                raise ValueError('Group-specific unconditional generator requires --checkpoint-map to export ALL condition groups')
        for start in range(0,samples_per_group,bs):
            n=min(bs,samples_per_group-start)
            y=torch.full((n,),cl,device=device,dtype=torch.long)
            v=torch.full((n,),vi,device=device,dtype=torch.long)
            if hasattr(model,'sample'):syn=model.sample(n,y,v,steps=sampling_steps)
            else:syn=model(torch.randn(n,group_info['model_config']['latent_dim'],device=device),y,v)
            for j in range(n):
                s=f'POOL_{kind}_{seed}_{cl}_{vi}_{start+j:06d}'
                out=images/f'{s}.png';_export_png(out,syn[j])
                records.append({'path':str(out),'split':'train','study_id':s,'subject_id':s,
                                'y':cl,'view':vi,'is_synthetic':True,'source_generator':kind,'seed':seed})
    out=output_dir/'candidate_manifest.csv';pd.DataFrame(records).to_csv(out,index=False)
    meta={'manifest':str(out),'samples':len(records),'groups':len(groups),'generator':kind,
          'purpose':'UNGATED samples for common-critic training ONLY; not augmentation'}
    write_json(output_dir/'candidate_export.json',meta)
    return meta
