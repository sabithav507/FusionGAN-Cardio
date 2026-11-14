"""Determinism, device handling, checkpoints, and diagnostics."""
from __future__ import annotations
import random, json, os, hashlib, contextlib
from pathlib import Path
import numpy as np
import torch

def set_seed(seed:int, deterministic:bool=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def get_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no CUDA GPU detected')
    return torch.device(name)

def amp_context(device, enabled=True):
    return torch.autocast(device_type='cuda', enabled=bool(enabled) and device.type=='cuda') if device.type=='cuda' else contextlib.nullcontext()

def grad_scaler(device, enabled=True):
    return torch.amp.GradScaler('cuda', enabled=bool(enabled) and device.type=='cuda')

def save_checkpoint(path, **payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    torch.save(payload,temp)
    os.replace(temp,path)

def load_checkpoint(path, device='cpu'):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def write_json(path, data):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:json.dump(data,f,indent=2,default=_to_json)

def _to_json(x):
    if isinstance(x,(np.integer,np.floating)):return x.item()
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    raise TypeError(str(type(x)))

def count_params(model):return sum(p.numel() for p in model.parameters())

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def jsonl_append(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a',encoding='utf-8') as f:f.write(json.dumps(record,default=_to_json)+'\n')

def freeze(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def require_file(path, role):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f'{role} checkpoint not found: {path}. Train it first or supply --{role.replace("_","-")}.')
    return Path(path)
