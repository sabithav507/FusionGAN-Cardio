#!/usr/bin/env python3
"""Repeat main FusionGAN and real-vs-augmented classifier on matching seeds.

Usage:
  python scripts/run_seed_study.py --config configs/full.yaml \
      --manifest data/processed/tmed_manifest.csv --structure outputs/structure.pt \
      --evaluator outputs/evaluator.pt --teacher outputs/teacher.pt \
      --outdir outputs/repeated_runs --seeds 7 21 42 77 123

Each run is complete and may take many GPU hours on full data. The script is
intentionally synchronous and never claims to perform work in the background.
"""
import argparse,subprocess,sys,json,csv
from pathlib import Path

def invoke(config,seed,*argv):
    command=[sys.executable,'-m','fusiongan_cardio','--config',config,'--set',f'project.seed={seed}',*map(str,argv)]
    print('+',' '.join(command),flush=True)
    subprocess.run(command,check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/full.yaml');p.add_argument('--manifest',required=True)
    p.add_argument('--structure',required=True);p.add_argument('--evaluator',required=True);p.add_argument('--teacher',required=True)
    p.add_argument('--outdir',required=True);p.add_argument('--task',default='tmed',choices=['tmed','echonet'])
    p.add_argument('--seeds',nargs='+',type=int,default=[7,21,42,77,123,256,512,1024,2048,4096]);a=p.parse_args()
    root=Path(a.outdir).resolve();root.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in a.seeds:
        base=root/f'seed_{seed}';base.mkdir(exist_ok=True)
        gan=base/'fusiongan.pt';balanced=base/'augmented';real_cl=base/'real_classifier.pt';aug_cl=base/'augmented_classifier.pt'
        invoke(a.config,seed,'train-gan','--manifest',a.manifest,'--task',a.task,'--structure',a.structure,'--evaluator',a.evaluator,'--out',gan)
        invoke(a.config,seed,'augment','--manifest',a.manifest,'--task',a.task,'--generator',gan,'--teacher',a.teacher,
               '--structure',a.structure,'--out',balanced)
        invoke(a.config,seed,'train-classifier','--manifest',a.manifest,'--task',a.task,'--out',real_cl,'--seed',seed)
        invoke(a.config,seed,'train-classifier','--manifest',balanced/'balanced_manifest.csv','--task',a.task,'--out',aug_cl,'--seed',seed)
        for model,ckpt in [('real_only',real_cl),('fusiongan',aug_cl)]:
            reportdir=base/model
            invoke(a.config,seed,'eval-classifier','--manifest',a.manifest,'--checkpoint',ckpt,'--out',reportdir)
            with open(reportdir/'test_metrics.json') as f:results=json.load(f)['image_level']
            for metric in ('macro_f1','balanced_accuracy','accuracy','auroc'):
                if results.get(metric) is not None:records.append({'seed':seed,'method':model,'metric_name':metric,'metric':results[metric]})
        with open(root/'all_metrics.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['seed','method','metric_name','metric']);writer.writeheader();writer.writerows(records)
        for metric in ('macro_f1','balanced_accuracy','accuracy','auroc'):
            with open(root/f'paired_{metric}.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['seed','method','metric']);writer.writeheader()
                writer.writerows({k:r[k] for k in ('seed','method','metric')} for r in records if r['metric_name']==metric)
    print(f'Finished {len(a.seeds)} seeds; metrics: {root / "all_metrics.csv"}')

if __name__=='__main__':main()
