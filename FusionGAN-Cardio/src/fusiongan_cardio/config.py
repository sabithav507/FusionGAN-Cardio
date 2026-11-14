"""Validated YAML configuration with optional dot-path overrides."""
from __future__ import annotations
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None, overrides=None):
    path = Path(path) if path else ROOT / 'configs' / 'full.yaml'
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Invalid config: {path}')
    for expression in overrides or []:
        if '=' not in expression:
            raise ValueError(f'Expected key=value, got {expression}')
        key, val = expression.split('=', 1)
        target = cfg
        keys = key.split('.')
        for sub in keys[:-1]:
            if sub not in target or not isinstance(target[sub], dict):
                raise KeyError(f'Unknown configuration key: {key}')
            target = target[sub]
        if keys[-1] not in target:
            raise KeyError(f'Unknown configuration key: {key}')
        target[keys[-1]] = yaml.safe_load(val)
    image_size = cfg['data']['image_size']
    if image_size < 32 or image_size > 512 or image_size & (image_size - 1):
        raise ValueError('image_size must be a power of two between 32 and 512')
    if not 0 < float(cfg['data']['balance_ratio']) <= 1:
        raise ValueError('balance_ratio must lie in (0,1]')
    if abs(sum(float(cfg['quality'][k]) for k in ('class_weight','realism_weight','diversity_weight'))-1)>1e-6:
        raise ValueError('Quality-gate weights must sum to 1')
    return cfg

def dump_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
