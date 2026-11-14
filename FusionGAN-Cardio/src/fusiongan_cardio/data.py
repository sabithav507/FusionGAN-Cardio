"""Strict dataset manifests, real-data adapters, preprocessing, and patient-level audit.

All manifests are CSV with path,split,subject_id,study_id,y,view,is_synthetic.
Paths may be absolute or relative to the manifest directory. Rows always represent
single still images; EchoNet videos are decoded BEFORE train/test use.
"""
from __future__ import annotations
import os, re, warnings, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

AS_LABELS = {'no_as':0,'none':0,'no as':0,'0':0, 'early_as':1,'early as':1,'early':1,'1':1,'mild':1,'mild_as':1,'mild_to_moderate_as':1,'mild_to_moderate':1,'mild_moderate':1,
             'significant_as':2,'significant as':2,'significant':2,'2':2,'moderate_or_worse':2,'moderate_as':2,'severe_as':2,'moderate':2,'severe':2}
VIEW_LABELS = {'plax':0,'psax':1,'a2c':2,'2ch':2,'2c':2,'a4c':3,'4ch':3,'4c':3,'other':4,
               '0':0,'1':1,'2':2,'3':3,'4':4}
EF_LABELS = {'reduced':0,'low':0,'0':0,'preserved':1,'normal':1,'1':1}
SPLITS = {'train':'train','training':'train','tr':'train','val':'val','valid':'val','validation':'val','dev':'val',
          'test':'test','testing':'test'}

def _label(value, mapping, name):
    if pd.isna(value):
        raise ValueError(f'Missing {name} label')
    key=str(value).strip().lower().replace('-','_')
    if key not in mapping:
        raise ValueError(f'Unknown {name} label {value!r}; expected {sorted(mapping)}')
    return mapping[key]

def _as_path(value, manifest):
    path=Path(str(value)).expanduser()
    return path if path.is_absolute() else (Path(manifest).resolve().parent/path).resolve()

def load_manifest(path, task='tmed', check_files=True, check_splits=True):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Manifest not found: {path}')
    df=pd.read_csv(path).fillna({'is_synthetic':False})
    required={'path','split','subject_id','study_id','y','view'}
    missing=required-set(df.columns)
    if missing:raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if len(df)==0:raise ValueError(f'Empty manifest: {path}')
    df['split']=df['split'].astype(str).str.strip().str.lower().map(SPLITS)
    if df['split'].isna().any():raise ValueError('Unknown train/val/test split')
    df['path']=df.path.map(lambda v:str(_as_path(v,path)))
    if 'mask_path' in df.columns:
        df['mask_path']=df.mask_path.map(lambda v:str(_as_path(v,path)) if pd.notna(v) and str(v).strip() else '')
    df['subject_id']=df.subject_id.astype(str)
    df['study_id']=df.study_id.astype(str)
    if (df.subject_id.str.strip()=='').any() or (df.study_id.str.strip()=='').any():
        raise ValueError('Empty subject_id or study_id')
    if task=='tmed': df['y']=df['y'].map(lambda y:_label(y,AS_LABELS,'AS'))
    elif task=='echonet':df['y']=df['y'].map(lambda y:_label(y,EF_LABELS,'EF'))
    elif task=='camus':
        df['y']=df['y'].astype(float).fillna(-1).astype(int)
    else:raise ValueError(f'Unknown task {task}')
    df['view']=df['view'].map(lambda v:_label(v,VIEW_LABELS,'view'))
    df['is_synthetic']=df['is_synthetic'].astype(str).str.lower().isin(['true','1','yes'])
    if df.path.duplicated().any():
        raise ValueError(f'Manifest repeats {int(df.path.duplicated().sum())} image paths')
    if check_files:
        absent=[p for p in df.path if not Path(p).is_file()]
        if absent:raise FileNotFoundError(f'{len(absent)} manifest images do not exist. First: {absent[0]}')
    if check_splits: audit_splits(df)
    return df.reset_index(drop=True)

def audit_splits(df):
    """Never allow one subject or study to contribute to multiple splits."""
    for key in ('subject_id','study_id'):
        counts=df.groupby(key)['split'].nunique()
        if (counts>1).any():
            bad=list(counts[counts>1].index[:6]);raise ValueError(f'LEAKAGE: {key} appears in multiple splits: {bad}')
    if df.loc[df['is_synthetic'],'split'].ne('train').any():
        raise ValueError('LEAKAGE: synthetic images appear outside train')
    return {'subjects':int(df.subject_id.nunique()),'studies':int(df.study_id.nunique()),'images':len(df),
            'splits':{str(k):int(v) for k,v in df['split'].value_counts().items()}}

def _to_gray(path):
    with Image.open(path) as raw:
        return raw.convert('L').copy()

def preprocess_image(img, size=256, mask=False):
    """Aspect-preserving resize with black letterbox; no horizontal reversal."""
    if not isinstance(img,Image.Image):img=Image.fromarray(np.asarray(img))
    if img.mode != 'L':img=img.convert('L')
    w,h=img.size
    if w<1 or h<1:raise ValueError('Empty image')
    scale=min(size/w,size/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    interp=Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    resized=img.resize((nw,nh),interp)
    dst=Image.new('L',(size,size),0)
    dst.paste(resized,((size-nw)//2,(size-nh)//2))
    return dst

class EchoImageDataset(Dataset):
    def __init__(self, frame, size=256, augment=False, with_masks=False):
        self.df=frame.reset_index(drop=True).copy()
        self.size=int(size);self.augment=augment;self.with_masks=with_masks
        if len(self.df)==0:raise ValueError('Dataset has zero rows')
    def __len__(self):return len(self.df)
    def __getitem__(self, index):
        row=self.df.iloc[index]
        image=preprocess_image(_to_gray(row.path),self.size)
        x=TF.to_tensor(image)
        if self.augment:
            angle=float(np.random.uniform(-10,10));scale=float(np.random.uniform(.9,1.1))
            translation=[int(np.random.uniform(-.05,.05)*self.size) for _ in range(2)]
            x=TF.affine(x,angle=angle,translate=translation,scale=scale,shear=[0.,0.], interpolation=InterpolationMode.BILINEAR)
            x=x*float(np.random.uniform(.9,1.1))+float(np.random.uniform(-.1,.1))
            x=x.clamp(0,1).pow(float(np.random.uniform(.9,1.1)))
            x=(x+torch.randn_like(x)*x*float(np.random.uniform(0,.05))).clamp(0,1)
        item={'image':x,'y':torch.tensor(int(row['y']),dtype=torch.long),
              'view':torch.tensor(int(row['view']),dtype=torch.long),
              'synthetic':torch.tensor(bool(row['is_synthetic']),dtype=torch.bool),
              'subject_id':str(row.subject_id),'study_id':str(row.study_id),'path':str(row.path)}
        if self.with_masks:
            path=row.get('mask_path','')
            if not path or not Path(path).exists():raise FileNotFoundError(f'CAMUS mask missing: {path}')
            # Masks are integer-valued (0 background, 1 LV, 2 myocardium, 3 atrium).
            with Image.open(path) as raw:
                mask=preprocess_image(raw.convert('L'),self.size,mask=True)
            item['mask']=torch.from_numpy(np.asarray(mask,dtype=np.uint8).astype(np.int64))
        if 'ef' in row and pd.notna(row.ef): item['ef']=torch.tensor(float(row.ef),dtype=torch.float32)
        if 'phase' in row:item['phase']=str(row.phase)
        return item

class RatioBatchSampler(Sampler):
    """Fixed real/synthetic fraction per batch, cycling both strata with replacement."""
    def __init__(self, df,batch_size,seed=42):
        self.real=np.flatnonzero(~df['is_synthetic'].to_numpy());self.syn=np.flatnonzero(df['is_synthetic'].to_numpy())
        self.bs=int(batch_size);self.seed=int(seed);self.epoch=0
        if not len(self.real):raise ValueError('No real training samples')
        frac=len(self.syn)/len(df);self.ns=max(1,round(self.bs*frac)) if len(self.syn) else 0
        if self.ns>=self.bs and len(self.real):self.ns=self.bs-1
        self.nr=self.bs-self.ns
    def __len__(self):return math.ceil((len(self.real)+len(self.syn))/self.bs)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);self.epoch+=1
        for _ in range(len(self)):
            real=rng.choice(self.real,self.nr,replace=len(self.real)<self.nr)
            syn=rng.choice(self.syn,self.ns,replace=len(self.syn)<self.ns) if self.ns else np.empty(0,dtype=int)
            group=np.concatenate([real,syn]);rng.shuffle(group);yield group.tolist()

def dataloader(df,cfg,split=None,augment=False,masks=False,batch_size=None,balanced=False,shuffle=None):
    if split:df=df[df['split']==split]
    if df.empty:raise ValueError(f'No images in split {split}')
    ds=EchoImageDataset(df,size=cfg['data']['image_size'],augment=augment,with_masks=masks)
    bs=batch_size or cfg['training']['batch_size'];workers=cfg['project'].get('num_workers',0)
    if balanced:
        return DataLoader(ds,batch_sampler=RatioBatchSampler(ds.df,bs,cfg['project']['seed']),num_workers=workers,pin_memory=torch.cuda.is_available())
    if shuffle is None:shuffle=split=='train'
    return DataLoader(ds,batch_size=bs,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),drop_last=False)

def _relpaths(manifest_path, records):
    # Store absolute canonical paths to avoid ambiguity when working directories change.
    frame=pd.DataFrame(records)
    Path(manifest_path).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(manifest_path,index=False)
    return str(manifest_path)

def prepare_tmed(metadata,output,source_root=None,seed=42,train_fraction=.60,val_fraction=.20,max_frames_per_view_study=2,
                 allow_study_only=False):
    """Convert TMED user metadata into auditable standard manifest.

    Metadata columns: image_path, study_id, patient_id (recommended), disease,
    view, split (recommended). Real TMED-2 has multiple studies per patient;
    if no patient_id, only official patient-separated split may be used unless
    explicit override allow_study_only=True, in which case grouping guarantee
    is study-level only and the output must not be described as patient-safe.
    """
    meta=pd.read_csv(metadata)
    required={'image_path','study_id','disease','view'}
    if not required.issubset(meta.columns):raise ValueError(f'TMED metadata missing {sorted(required-set(meta.columns))}')
    if 'patient_id' not in meta and 'split' not in meta and not allow_study_only:
        raise ValueError('TMED requires patient_id for a new split; official release has 599 studies from 577 patients. Supply a provided patient-level split or --allow-study-only (not patient-safe).')
    root=Path(source_root).resolve() if source_root else Path(metadata).resolve().parent
    meta['study_id']=meta.study_id.astype(str)
    meta['subject_id']=meta.patient_id.astype(str) if 'patient_id' in meta else meta.study_id
    if 'split' in meta:
        meta['split']=meta['split'].astype(str).str.lower().map(SPLITS)
        if meta['split'].isna().any():raise ValueError('TMED metadata has unknown split')
    else:
        # Partition at patient level. All studies for a patient remain together.
        subject_ids=sorted(meta.subject_id.unique());rng=np.random.default_rng(seed);rng.shuffle(subject_ids)
        n=len(subject_ids);a=round(n*train_fraction);b=round(n*(train_fraction+val_fraction))
        lookup={str(k):('train' if i<a else ('val' if i<b else 'test')) for i,k in enumerate(subject_ids)}
        meta['split']=meta.subject_id.map(lookup)
    meta['y']=meta.disease.map(lambda v:_label(v,AS_LABELS,'AS'))
    meta['view']=meta['view'].map(lambda v:_label(v,VIEW_LABELS,'view'))
    meta['path']=meta.image_path.map(lambda s:str((root/str(s)).resolve()))
    if not all(Path(p).is_file() for p in meta.path):
        p=next(p for p in meta.path if not Path(p).is_file());raise FileNotFoundError(f'Missing TMED image {p}')
    # Deterministic selection (not arbitrary filename order) of at most two per view/study.
    meta=meta.sort_values(['study_id','view','path']).groupby(['study_id','view'],group_keys=False).head(max_frames_per_view_study)
    records=meta[['path','split','subject_id','study_id','y','view']].copy();records['is_synthetic']=False
    audit_splits(records)
    return _relpaths(output,records)

def prepare_echonet(root,output,images_dir,frames_per_video=5,max_videos=None,skip_invalid=False):
    """Decode the official EchoNet-Dynamic FileList.csv and Videos/*.avi."""
    import cv2
    root=Path(root).resolve();df=pd.read_csv(root/'FileList.csv')
    for col in ('FileName','EF','Split'):
        if col not in df:raise ValueError(f'EchoNet FileList.csv missing {col}')
    if max_videos:df=df.head(max_videos)
    dest=Path(images_dir).resolve();dest.mkdir(parents=True,exist_ok=True)
    records=[];rejects=[]
    for row in df.itertuples(index=False):
        fname=str(getattr(row,'FileName'));name=fname if Path(fname).suffix else fname+'.avi'
        safe=Path(name).name;video=root/'Videos'/safe
        if not video.is_file():
            if skip_invalid:rejects.append(str(video));continue
            raise FileNotFoundError(video)
        cap=cv2.VideoCapture(str(video))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total<1:
            cap.release()
            if skip_invalid:rejects.append(str(video));continue
            raise ValueError(f'Video unreadable: {video}')
        indices=np.unique(np.round(np.linspace(0,total-1,frames_per_video)).astype(int))
        frames=[]
        for ix in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix));ok,arr=cap.read()
            if not ok:continue
            gray=cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
            if gray.max()-gray.min()<10:continue
            frames.append((int(ix),gray))
        cap.release()
        if not frames:
            rejects.append(str(video));continue
        ef=float(getattr(row,'EF'))
        if not np.isfinite(ef):rejects.append(str(video));continue
        split=SPLITS.get(str(getattr(row,'Split')).lower())
        if split is None:raise ValueError(f'Unknown EchoNet split in {video}')
        study=Path(safe).stem
        for ix,gray in frames:
            out=dest/f'{study}_f{ix:04d}.png'
            Image.fromarray(gray).save(out)
            records.append({'path':str(out),'split':split,'subject_id':study,'study_id':study,
                            'y':int(ef>=50.0),'view':3,'ef':ef,'frame_index':ix,'is_synthetic':False})
    if not records:raise ValueError('No readable EchoNet frames')
    frame=pd.DataFrame(records);audit_splits(frame)
    if rejects:Path(str(output)+'.rejected.txt').write_text('\n'.join(rejects),encoding='utf-8')
    return _relpaths(output,frame)

def _read_medical_volume(path):
    path=Path(path)
    if str(path).endswith(('.nii','.nii.gz')):
        try:import nibabel as nib
        except ImportError as e:raise ImportError('Install nibabel for CAMUS NIfTI: pip install nibabel') from e
        return np.squeeze(np.asarray(nib.load(str(path)).dataobj))
    if str(path).endswith(('.mhd','.mha')):
        try:import SimpleITK as sitk
        except ImportError as e:raise ImportError('Install SimpleITK to read original CAMUS .mhd: pip install SimpleITK') from e
        return np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
    raise ValueError(f'Unsupported CAMUS image {path}')

def _uint8_array(array,mask=False):
    arr=np.asarray(array)
    if arr.ndim!=2:raise ValueError(f'Expected 2D CAMUS endpoint, got {arr.shape}')
    if mask:
        out=np.rint(arr).astype(np.uint8)
        if out.max()>3:raise ValueError('CAMUS mask labels outside 0..3 (check input segmentation convention)')
        return out
    arr=np.nan_to_num(arr.astype(np.float32),nan=0,posinf=0,neginf=0)
    if arr.min()<0 or arr.max()>255:
        a,b=np.percentile(arr,[0,99.9]);arr=(arr-a)/(b-a+1e-7)*255
    return np.clip(arr,0,255).astype(np.uint8)

def prepare_camus(root,output,images_dir,seed=42,test_fraction=.20):
    """Export official CAMUS endpoint images and GT masks from .nii.gz or .mhd.

    If an official split directory exists, it is used when readable. Otherwise a
    deterministically seeded patient-level 400/100 split is created, as the
    manuscript states. This is NOT claimed to reproduce unavailable IDs.
    """
    root=Path(root).resolve();dest=Path(images_dir).resolve();dest.mkdir(parents=True,exist_ok=True)
    parents=list(root.rglob('patient????'))
    if not parents:raise ValueError('No patient???? folders found under CAMUS root')
    parents=sorted(set(p for p in parents if p.is_dir()))
    patient_names=[p.name for p in parents]
    # Use official subgroup split files if present: train remains 400, val+test form held-out 100.
    official={}
    for f in root.rglob('subgroup_*.txt'):
        lo=f.name.lower();split=('train' if 'train' in lo else ('test' if 'test' in lo or 'val' in lo else None))
        if split:
            for name in re.findall(r'patient\d+',f.read_text(errors='ignore'),re.I):official[name.lower()]=split
    if official and not all(n.lower() in official for n in patient_names):
        warnings.warn('Incomplete CAMUS official split files; using reproducible seeded patient split')
        official={}
    if not official:
        rng=np.random.default_rng(seed);names=patient_names.copy();rng.shuffle(names)
        n_train=round(len(names)*(1-test_fraction));official={name:('train' if i<n_train else 'test') for i,name in enumerate(names)}
    records=[]
    for patient in parents:
        for view_token,view_int in [('2CH',2),('4CH',3)]:
            for phase in ['ED','ES']:
                matches=list(patient.glob(f'{patient.name}_{view_token}_{phase}.nii.gz'))+list(patient.glob(f'{patient.name}_{view_token}_{phase}.mhd'))
                if not matches:continue
                inp=matches[0];extension='.nii.gz' if str(inp).endswith('.nii.gz') else '.mhd'
                mask=patient/f'{patient.name}_{view_token}_{phase}_gt{extension}'
                if not mask.exists():continue
                image_data=_uint8_array(_read_medical_volume(inp));mask_data=_uint8_array(_read_medical_volume(mask),mask=True)
                image_path=dest/f'{patient.name}_{view_token}_{phase}.png';mask_path=dest/f'{patient.name}_{view_token}_{phase}_mask.png'
                Image.fromarray(image_data).save(image_path);Image.fromarray(mask_data).save(mask_path)
                info_file=patient/f'Info_{view_token}.cfg'
                ef_value=None
                if info_file.is_file():
                    info_text=info_file.read_text(errors='ignore')
                    for pattern in (r'(?im)^\s*LVef\s*[:=]\s*([0-9.]+)',r'(?im)^\s*EF\s*[:=]\s*([0-9.]+)'):
                        match=re.search(pattern,info_text)
                        if match:
                            ef_value=float(match.group(1));break
                records.append({'path':str(image_path),'mask_path':str(mask_path),'split':official[patient.name.lower()],
                                'subject_id':patient.name,'study_id':f'{patient.name}_{view_token}',
                                'y':-1,'view':view_int,'phase':phase,'ef':ef_value,'is_synthetic':False})
    if not records:raise ValueError('No CAMUS ED/ES images AND masks found (check raw file names)')
    frame=pd.DataFrame(records);audit_splits(frame)
    return _relpaths(output,frame)
