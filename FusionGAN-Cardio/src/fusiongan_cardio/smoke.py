"""Generate wholly synthetic, non-clinical test fixtures and exercise the full pipeline.
These artificial geometric test images must NEVER be used in a paper as clinical data.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

def create_smoke_data(root,image_size=40,seed=42):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    generated={}
    for task in ('tmed','echonet','camus'):
        rows=[];folder=root/task;folder.mkdir(parents=True,exist_ok=True)
        if task=='tmed':
            combinations=[(0,2,6),(1,2,3),(2,3,2)]
            samples=[]
            for y,v,count in combinations:samples.extend([('train',y,v)]*count)
            samples += [('val',i%3,2+i%2) for i in range(6)]
            samples += [('test',i%3,2+i%2) for i in range(6)]
        elif task=='echonet':
            samples=[('train',i%2,3) for i in range(12)]+[('val',i%2,3) for i in range(4)]+[('test',i%2,3) for i in range(4)]
        else:
            samples=[('train',-1,2+i%2) for i in range(12)]+[('test',-1,2+i%2) for i in range(4)]
        for i,(split,y,v) in enumerate(samples):
            # Gray ultrasound-like synthetic triangle + elliptical chamber + image noise.
            arr=rng.normal(18,7,(image_size,image_size)).clip(0,255).astype(np.uint8)
            image=Image.fromarray(arr).convert('L');overlay=Image.new('L',(image_size,image_size),0)
            draw=ImageDraw.Draw(overlay)
            draw.polygon([(image_size//2,2),(2,image_size-2),(image_size-2,image_size-2)],fill=100)
            cx=image_size//2+((i%3)-1);cy=image_size//2
            r=image_size//5
            draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=30)
            image=Image.blend(image,overlay,0.7)
            path=folder/f'fixture_{split}_{i:03d}.png';image.save(path)
            row={'path':str(path),'split':split,'subject_id':f'{task}_fixture_{i}',
                 'study_id':f'{task}_fixture_{i}','y':y,'view':v,'is_synthetic':False}
            if task=='echonet':row['ef']=35 if y==0 else 65
            if task=='camus':
                mask=Image.new('L',(image_size,image_size),0);pen=ImageDraw.Draw(mask)
                pen.ellipse((cx-r-2,cy-r-2,cx+r+2,cy+r+2),fill=2)
                pen.ellipse((cx-r,cy-r,cx+r,cy+r),fill=1)
                pen.ellipse((cx-r,cy+r//2,cx+r,cy+r*2),fill=3)
                mask_path=folder/f'fixture_{split}_{i:03d}_mask.png';mask.save(mask_path)
                row['mask_path']=str(mask_path);row['phase']='ED' if i%2 else 'ES'
            rows.append(row)
        out=root/f'{task}.csv';pd.DataFrame(rows).to_csv(out,index=False);generated[task]=str(out)
    return generated

def run_smoke(workdir):
    """Runs real forward/backprop + train/evaluate/augment with synthetic fixtures."""
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    from .config import load_config
    from .train import fit_structure,fit_evaluator,fit_probe,fit_classifier,fit_gan,evaluate_classifier
    from .quality import fit_balanced_manifest
    from .evaluate import evaluate_generator,evaluate_anatomy
    from .utils import write_json
    root=Path(workdir).resolve();files=create_smoke_data(root/'fixtures')
    cfg=load_config(Path(__file__).resolve().parents[2]/'configs'/'smoke.yaml')
    ckdir=root/'checkpoints';ckdir.mkdir(parents=True,exist_ok=True)
    structure=fit_structure(cfg,files['camus'],files['echonet'],ckdir/'structure.pt')
    evaluator=fit_evaluator(cfg,files['tmed'],ckdir/'evaluator.pt')
    probe=fit_probe(cfg,files['camus'],ckdir/'probe.pt')
    teacher=fit_classifier(cfg,files['tmed'],ckdir/'teacher.pt',teacher=True,architecture='tiny')
    gan=fit_gan(cfg,files['tmed'],structure['checkpoint'],evaluator['checkpoint'],ckdir/'fusiongan.pt')
    augmented=fit_balanced_manifest(cfg,files['tmed'],gan['checkpoint'],teacher['checkpoint'],
                    structure['checkpoint'],root/'augmented')
    classifier=fit_classifier(cfg,augmented['manifest'],ckdir/'classifier.pt',architecture='tiny')
    test=evaluate_classifier(cfg,augmented['manifest'],classifier['checkpoint'],output=root/'evaluation')
    gen_eval=evaluate_generator(cfg,files['tmed'],gan['checkpoint'],evaluator['checkpoint'],
                               root/'generative_eval',split='test',max_images=6,lpips_pairs=2)
    anatomy=evaluate_anatomy(cfg,files['camus'],probe['checkpoint'],structure['checkpoint'],
                            gan['checkpoint'],root/'anatomy',n_generated=2)
    report={'fixtures':'synthetic_nonclinical_only','aux_structure':structure,'aux_evaluator':evaluator,
            'aux_probe':probe,'teacher':teacher,'gan':gan,'augmentation':augmented,
            'classifier':classifier,'test_metrics':test,'generative_metrics':gen_eval,'anatomy':anatomy}
    write_json(root/'smoke_result.json',report)
    return {'status':'PASS','report':str(root/'smoke_result.json'),'zip_ready':True,
            'test_macro_f1':test['image_level']['macro_f1'],'generated_count':augmented['n_synthetic']}
