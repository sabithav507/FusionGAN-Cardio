"""Independent generative metrics, reference-matched CAMUS structural validation,
classifier prediction outputs and study-level statistics.
"""
from __future__ import annotations
from pathlib import Path
import warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from .data import load_manifest,dataloader
from .train import load_gan,load_evaluator,load_structure,load_probe
from .baseline_train import load_baseline
from .utils import get_device,write_json,set_seed
from .metrics import (frechet_distance,kernel_inception_distance,generative_precision_recall,
                      dice,hd95)

@torch.no_grad()
def evaluate_generator(cfg,manifest,checkpoint,evaluator_checkpoint,output,task='tmed',kind='fusion',
                       split='test',max_images=1000,seed=42,lpips_pairs=32,
                       include_inception=False,radimagenet_weights=None):
    """All evaluation real images come from the heldout split; generated images
    are conditioned on heldout class/view frequencies but never memorize their pixels.
    """
    set_seed(seed);device=get_device(cfg['project']['device'])
    df=load_manifest(manifest,task)
    if kind=='fusion':gen,_,ck=load_gan(checkpoint,device)
    else:gen,ck=load_baseline(checkpoint,device)
    evaluator,ev_info=load_evaluator(evaluator_checkpoint,device)
    alternatives={}
    if include_inception or radimagenet_weights:
        from .models.evaluation import IndependentFeatureEncoder
        if include_inception:alternatives['inception_v3']=IndependentFeatureEncoder('inception_v3').to(device).eval()
        if radimagenet_weights:alternatives['radimagenet_resnet50']=IndependentFeatureEncoder('radimagenet_resnet50',radimagenet_weights).to(device).eval()
    alternative_features={name:{'real':[],'synthetic':[]} for name in alternatives}
    subset=df[(df['split']==split)&(~df['is_synthetic'])].copy()
    if ck.get('group_class') is not None:
        subset=subset[(subset['y']==ck['group_class'])&(subset['view']==ck['group_view'])]
    if subset.empty:raise ValueError('No held-out real evaluation samples for generator conditions')
    if max_images:subset=subset.sample(n=min(max_images,len(subset)),random_state=seed).sort_index()
    loader=dataloader(subset,cfg,batch_size=cfg['benchmark']['eval_batch_size'],shuffle=False)
    real_emb=[];fake_emb=[];pairs=[];details=[]
    for b in loader:
        x=b['image'].to(device);y=b['y'].to(device);v=b['view'].to(device)
        if hasattr(gen,'sample'):syn=gen.sample(len(x),y,v,steps=50)
        else:syn=gen(torch.randn(len(x),ck['model_config']['latent_dim'],device=device),y,v)
        real_emb.append(evaluator(x).float().cpu().numpy());fake_emb.append(evaluator(syn).float().cpu().numpy())
        for name,network in alternatives.items():
            alternative_features[name]['real'].append(network(x).float().cpu().numpy())
            alternative_features[name]['synthetic'].append(network(syn).float().cpu().numpy())
        if len(pairs)<lpips_pairs:
            pairs.extend([syn[j].detach().cpu() for j in range(min(len(syn),lpips_pairs-len(pairs)))])
        details.extend([{'view':int(v[i]),'y':int(y[i])} for i in range(len(y))])
    x=np.concatenate(real_emb);g=np.concatenate(fake_emb)
    report={'model':kind,'task':task,'split':split,'real_count':len(x),'synthetic_count':len(g),
            'feature_encoder_source':ev_info.get('feature_source'),'fid_feature_dim':x.shape[1],
            'medical_feature_frechet_distance':frechet_distance(x,g)}
    report['kid_mean'],report['kid_sd']=kernel_inception_distance(x,g,subsets=10,seed=seed)
    for name,features in alternative_features.items():
        report[f'fid_{name}']=frechet_distance(np.concatenate(features['real']),np.concatenate(features['synthetic']))
    try:
        p,r=generative_precision_recall(x,g,k=cfg['benchmark'].get('pr_k',3))
        report['generative_precision']=p;report['generative_recall']=r
    except ValueError as ex:report['generative_precision']=None;report['generative_recall']=None;report['precision_recall_note']=str(ex)
    # Per disease-view Fréchet where there are sufficient examples.
    details=pd.DataFrame(details)
    report['by_condition']={}
    for key,ids in details.groupby(['y','view']).indices.items():
        ids=np.asarray(ids)
        if len(ids)>=2:report['by_condition'][f'{key[0]}:{key[1]}']=frechet_distance(x[ids],g[ids])
    # Optional independent perceptual distance, without substitution if dependencies absent.
    try:
        import lpips
        if len(pairs)>=2:
            scorer=lpips.LPIPS(net='alex',pretrained=True).to(device).eval()
            pair_a=torch.stack(pairs[::2]);pair_b=torch.stack(pairs[1::2]);n=min(len(pair_a),len(pair_b))
            report['lpips_pair_diversity']=float(scorer(pair_a[:n].to(device).repeat(1,3,1,1)*2-1,
                                                    pair_b[:n].to(device).repeat(1,3,1,1)*2-1).mean())
        else:report['lpips_pair_diversity']=None
    except (ImportError,RuntimeError) as exc:
        report['lpips_pair_diversity']=None;report['lpips_note']=f'Optional lpips unavailable: {str(exc)[:180]}'
    try:
        from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as ms_ssim
        if len(pairs)>=2 and pairs[0].shape[-1]>=160:
            pa=torch.stack(pairs[::2]);pb=torch.stack(pairs[1::2]);n=min(len(pa),len(pb))
            report['ms_ssim_pair_similarity']=float(ms_ssim(pa[:n].to(device),pb[:n].to(device),data_range=1.))
        else:report['ms_ssim_pair_similarity']=None
    except (ImportError,RuntimeError,ValueError) as exc:
        report['ms_ssim_pair_similarity']=None;report['ms_ssim_note']=str(exc)[:200]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'generative_metrics.json',report)
    return report

def _shape_info(binary):
    coords=np.argwhere(binary)
    if not len(coords):return np.array([binary.shape[0]/2,binary.shape[1]/2]),1.,0.
    center=coords.mean(axis=0)
    if len(coords)<3:return center,max(1.,np.sqrt(len(coords))),0.
    centered=coords-center
    eigenvalues,eigenvectors=np.linalg.eigh(np.cov(centered,rowvar=False));direction=eigenvectors[:,np.argmax(eigenvalues)]
    angle=float(np.arctan2(direction[0],direction[1])*180/np.pi)
    return center,max(1.,np.sqrt(len(coords))),angle

def align_reference_mask(reference,target_prediction):
    """Rigid isotropic matching using LV centroid, area-derived scale and long axis.
    This is a *proxy* to measure plausible anatomy; NOT independent truth.
    """
    import cv2
    ref=np.asarray(reference,dtype=np.uint8);pred=np.asarray(target_prediction,dtype=np.uint8)
    a,sa,angle_a=_shape_info(ref==1);b,sb,angle_b=_shape_info(pred==1)
    # PCA axes have 180-degree ambiguity; choose smallest magnitude rotation.
    angle=(angle_b-angle_a+90)%180-90
    rotation=cv2.getRotationMatrix2D((float(a[1]),float(a[0])),angle,float(np.clip(sb/sa,.5,2)))
    rotation[0,2]+=float(b[1]-a[1]);rotation[1,2]+=float(b[0]-a[0])
    return cv2.warpAffine(ref,rotation,(ref.shape[1],ref.shape[0]),flags=cv2.INTER_NEAREST,borderValue=0)

@torch.no_grad()
def evaluate_anatomy(cfg,camus_manifest,probe_checkpoint,structure_checkpoint,generator_checkpoint,output,
                     n_generated=100,seed=42):
    """Heldout real expert mask metrics + synthetic/reference proxy metrics, segregated.
    Generator must support y/view A2C/A4C; CAMUS has no TMED disease labels.
    """
    device=get_device(cfg['project']['device']);set_seed(seed)
    df=load_manifest(camus_manifest,'camus')
    heldout=df[df['split']=='test'].copy()
    if heldout.empty:raise ValueError('CAMUS test partition required; cannot use training masks for validation')
    probe,_=load_probe(probe_checkpoint,device)
    structure,_=load_structure(structure_checkpoint,device)
    gen,_,gck=load_gan(generator_checkpoint,device)
    if gck['num_views']<4:raise ValueError('Generator must include A2C and A4C conditions')
    loader=dataloader(heldout,cfg,masks=True,batch_size=cfg['benchmark']['eval_batch_size'],shuffle=False)
    real_records=[];feature_ref=[];mask_ref=[];view_ref=[];phase_ref=[]
    for b in loader:
        x=b['image'].to(device);true=b['mask'].cpu().numpy()
        pred=probe(x).argmax(1).cpu().numpy()
        emb=structure(x).float().cpu()
        for i in range(len(x)):
            row={'view':int(b['view'][i]),'phase':b.get('phase',['unknown']*len(x))[i]}
            for label in [1,2,3]:
                row[f'dice_{label}']=dice(pred[i]==label,true[i]==label)
                row[f'hd95_{label}_pixels']=hd95(pred[i]==label,true[i]==label)
            real_records.append(row);feature_ref.append(emb[i]);mask_ref.append(true[i]);view_ref.append(int(b['view'][i]));phase_ref.append(row['phase'])
    bank=torch.stack(feature_ref).to(device)
    rows=[]
    for index in range(n_generated):
        v=[2,3][index%2];y=index%gck['num_classes'];V=torch.tensor([v],device=device);Y=torch.tensor([y],device=device)
        fake=gen(torch.randn(1,gck['model_config']['latent_dim'],device=device),Y,V)
        mask_pred=probe(fake).argmax(1).squeeze().cpu().numpy()
        fake_emb=structure(fake).float()
        eligible=[i for i,real_view in enumerate(view_ref) if real_view==v]
        if not eligible:continue
        nearest=torch.cdist(fake_emb,bank[eligible]).argmin().item()
        ref=mask_ref[eligible[nearest]]
        aligned=align_reference_mask(ref,mask_pred)
        row={'condition_y':y,'view':v,'reference_phase':phase_ref[eligible[nearest]]}
        for label in [1,2,3]:
            row[f'proxy_dice_{label}']=dice(mask_pred==label,aligned==label)
            row[f'proxy_hd95_{label}_pixels']=hd95(mask_pred==label,aligned==label)
        rows.append(row)
    realdf=pd.DataFrame(real_records);syn=pd.DataFrame(rows)
    summary={'n_real_heldout':len(realdf),'n_synthetic':len(syn),
             'real_expert_dice':{str(i):float(realdf[f'dice_{i}'].mean()) for i in [1,2,3]},
             'synthetic_reference_matched_proxy_dice':{str(i):float(syn[f'proxy_dice_{i}'].mean()) if len(syn) else None for i in [1,2,3]},
             'warning':'Synthetic reference-matched Dice uses nearest real expert mask aligned using the predicted LV; it is NOT expert ground truth or proof of clinical plausibility.',
             'distances':'HD95 reported in resized pixel units; not physically calibrated millimetres'}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    realdf.to_csv(out/'real_camus_probe.csv',index=False);syn.to_csv(out/'synthetic_reference_proxy.csv',index=False)
    write_json(out/'anatomy_summary.json',summary)
    return summary


@torch.no_grad()
def evaluate_external_ef(cfg,camus_manifest,echonet_classifier_checkpoint,output):
    """External static-frame EF proxy transfer from EchoNet to heldout CAMUS.
    Only meaningful when CAMUS EF metadata are present; not video-level EF estimation.
    """
    from scipy.stats import spearmanr
    from .train import load_classifier
    from .metrics import classify_metrics
    device=get_device(cfg['project']['device'])
    classifier,info=load_classifier(echonet_classifier_checkpoint,device)
    if info['task']!='echonet' or info['num_classes']!=2:
        raise ValueError('External EF requires a binary EchoNet classifier checkpoint')
    df=load_manifest(camus_manifest,'camus');held=df[(df['split']=='test') & df['ef'].notna()].copy()
    if held.empty:raise ValueError('Heldout CAMUS EF metadata absent: cannot perform external EF evaluation')
    loader=dataloader(held,cfg,batch_size=cfg['benchmark']['eval_batch_size'],shuffle=False)
    rows=[]
    for b in loader:
        x=b['image'].to(device);p=F.softmax(classifier(x),dim=1).cpu().numpy()
        for i in range(len(p)):
            rows.append({'subject_id':b['subject_id'][i],'study_id':b['study_id'][i],
                         'ef':float(b['ef'][i]),'p_0':float(p[i,0]),'p_1':float(p[i,1])})
    values=pd.DataFrame(rows)
    # One patient contributes multiple ED/ES/A2C/A4C frames; aggregate per subject.
    by_subject=values.groupby('subject_id')[['ef','p_0','p_1']].mean()
    target=(by_subject.ef.to_numpy()>=50).astype(int)
    probs=by_subject[['p_0','p_1']].to_numpy()
    report={'task':'external CAMUS static-frame reduced/preserved EF proxy',
            'subjects':len(by_subject),'patient_level':classify_metrics(target,probs),
            'positive_preserved_probability_vs_EF_spearman':float(spearmanr(by_subject.ef,by_subject.p_1).statistic),
            'caveat':'Neither CAMUS ED/ES stills nor EchoNet five isolated stills constitute temporal EF measurements.'}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    values.to_csv(out/'external_ef_frames.csv',index=False)
    by_subject.to_csv(out/'external_ef_patients.csv')
    write_json(out/'external_ef_summary.json',report)
    return report
