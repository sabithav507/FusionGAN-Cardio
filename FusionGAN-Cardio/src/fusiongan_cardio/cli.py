"""fusiongan: one command-line interface for every research pipeline stage."""
from __future__ import annotations
import argparse, json
from .config import load_config

def build_parser():
    p=argparse.ArgumentParser(prog='fusiongan',description='FusionGAN-Cardio research implementation (NOT clinical software)')
    p.add_argument('--config',default=None,help='YAML config, default configs/full.yaml')
    p.add_argument('--set',action='append',default=[],metavar='KEY=VALUE',help='Override config e.g. training.epochs=2')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare-tmed',help='Build a TMED manifest from image-level metadata CSV')
    a.add_argument('--metadata',required=True);a.add_argument('--out',required=True);a.add_argument('--root')
    a.add_argument('--allow-study-only',action='store_true')
    a=s.add_parser('prepare-echonet',help='Extract five frames per official EchoNet video')
    a.add_argument('--root',required=True);a.add_argument('--out',required=True);a.add_argument('--images',required=True)
    a.add_argument('--max-videos',type=int);a.add_argument('--skip-invalid',action='store_true')
    a=s.add_parser('prepare-camus',help='Convert official CAMUS .nii.gz/.mhd ED/ES + masks')
    a.add_argument('--root',required=True);a.add_argument('--out',required=True);a.add_argument('--images',required=True)
    a=s.add_parser('audit',help='Check group-leakage and synthetic train-only constraints')
    a.add_argument('--manifest',required=True);a.add_argument('--task',choices=['tmed','echonet','camus'],default='tmed')
    a=s.add_parser('train-structure',help='Train CAMUS/EchoNet frozen ResNet-18 anatomical encoder')
    a.add_argument('--camus',required=True);a.add_argument('--echonet',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('train-probe',help='Train held-out CAMUS U-Net structural probe')
    a.add_argument('--camus',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('train-evaluator',help='Train independent train-only TMED SimCLR encoder')
    a.add_argument('--manifest',required=True);a.add_argument('--out',required=True)
    a.add_argument('--unlabeled-manifest',help='Optional TMED-2 unlabelled images manifest: path,study_id,subject_id')
    a=s.add_parser('train-teacher',help='Train independent DenseNet-121 real-only class-confidence teacher')
    a.add_argument('--manifest',required=True);a.add_argument('--task',choices=['tmed','echonet'],default='tmed');a.add_argument('--out',required=True)
    a=s.add_parser('train-gan',help='Train core FusionGAN with all six generator objectives')
    a.add_argument('--manifest',required=True);a.add_argument('--task',choices=['tmed','echonet'],default='tmed')
    a.add_argument('--structure',required=True);a.add_argument('--evaluator',required=True)
    a.add_argument('--out',required=True);a.add_argument('--resume')
    a=s.add_parser('train-baseline',help='Train GAN or diffusion reference comparators')
    a.add_argument('--name',choices=['dcgan','cgan','wgangp','stylegan2ada_lite','vitgan_lite','ddpm','ldm'],required=True)
    a.add_argument('--manifest',required=True);a.add_argument('--task',choices=['tmed','echonet'],default='tmed')
    a.add_argument('--evaluator',required=True);a.add_argument('--out',required=True)
    a.add_argument('--group-class',type=int);a.add_argument('--group-view',type=int)
    a=s.add_parser('sample-candidates',help='Export ungated candidate pools for fitting the common realism critic')
    a.add_argument('--manifest',required=True);a.add_argument('--generator',required=True);a.add_argument('--out',required=True)
    a.add_argument('--task',choices=['tmed','echonet'],default='tmed');a.add_argument('--kind',choices=['fusion','baseline'],default='fusion')
    a.add_argument('--n-per-group',type=int,default=100);a.add_argument('--checkpoint-map')
    a.add_argument('--seed',type=int);a.add_argument('--sampling-steps',type=int,default=50)
    a=s.add_parser('train-critic',help='Train a common ResNet-18 realism critic for fair comparisons')
    a.add_argument('--real',required=True);a.add_argument('--candidates',nargs='+',required=True)
    a.add_argument('--out',required=True);a.add_argument('--task',choices=['tmed','echonet'],default='tmed')
    a=s.add_parser('augment',help='Quality score candidates and construct train-only balanced manifest')
    a.add_argument('--manifest',required=True);a.add_argument('--generator',required=True)
    a.add_argument('--teacher',required=True);a.add_argument('--structure',required=True)
    a.add_argument('--out',required=True);a.add_argument('--task',choices=['tmed','echonet'],default='tmed')
    a.add_argument('--kind',choices=['fusion','baseline'],default='fusion');a.add_argument('--critic')
    a.add_argument('--without-gating',action='store_true');a.add_argument('--checkpoint-map',help='JSON mapping y:view -> baseline checkpoint')
    a.add_argument('--seed',type=int);a.add_argument('--sampling-steps',type=int,default=50)
    a=s.add_parser('train-classifier',help='Train fixed EfficientNet-B0 on real or quality-controlled augmented data')
    a.add_argument('--manifest',required=True);a.add_argument('--task',choices=['tmed','echonet'],default='tmed')
    a.add_argument('--arch',choices=['efficientnet_b0','resnet50','vit_b_16','tiny'])
    a.add_argument('--out',required=True);a.add_argument('--seed',type=int)
    a=s.add_parser('eval-classifier',help='Real-only test/val classification + study aggregation')
    a.add_argument('--manifest',required=True);a.add_argument('--checkpoint',required=True)
    a.add_argument('--out',required=True);a.add_argument('--split',choices=['test','val'],default='test')
    a=s.add_parser('eval-generator',help='Medical Fréchet/KID/precision/recall/perceptual metrics')
    a.add_argument('--manifest',required=True);a.add_argument('--checkpoint',required=True)
    a.add_argument('--evaluator',required=True);a.add_argument('--out',required=True)
    a.add_argument('--task',choices=['tmed','echonet'],default='tmed');a.add_argument('--kind',choices=['fusion','baseline'],default='fusion')
    a.add_argument('--split',choices=['val','test'],default='test');a.add_argument('--max-images',type=int,default=1000)
    a.add_argument('--inception',action='store_true',help='Optional independent ImageNet Inception-v3 FID')
    a.add_argument('--radimagenet-weights',help='External pretrained RadImageNet ResNet-50 checkpoint')
    a=s.add_parser('eval-anatomy',help='CAMUS real expert-mask evaluation plus synthetic proxy matching')
    a.add_argument('--camus',required=True);a.add_argument('--probe',required=True);a.add_argument('--structure',required=True)
    a.add_argument('--generator',required=True);a.add_argument('--out',required=True);a.add_argument('--n-generated',type=int,default=100)
    a=s.add_parser('eval-external-ef',help='EchoNet-trained static EF proxy evaluated on heldout CAMUS')
    a.add_argument('--camus',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('compare-seeds',help='Exact Wilcoxon, 95%% CIs, Holm-adjusted paired tests')
    a.add_argument('--csv',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('bootstrap',help='Paired study-level bootstrap predictions')
    a.add_argument('--a',required=True);a.add_argument('--b',required=True);a.add_argument('--out',required=True)
    a.add_argument('--metric',choices=['macro_f1','balanced_accuracy'],default='macro_f1')
    a.add_argument('--reps',type=int,default=10000)
    a=s.add_parser('profile',help='Measure generator/classifier inference time and peak CUDA memory')
    a.add_argument('--checkpoint',required=True);a.add_argument('--kind',choices=['fusion','baseline','classifier'],default='fusion')
    a.add_argument('--out',required=True);a.add_argument('--batch-size',type=int,default=1)
    a.add_argument('--warmup',type=int,default=50);a.add_argument('--runs',type=int,default=1000)
    a.add_argument('--sampling-steps',type=int,default=50)
    a=s.add_parser('plot-curves');a.add_argument('--log',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('plot-grid');a.add_argument('--manifest',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('plot-balance',help='600-dpi real-vs-synthetic condition histogram')
    a.add_argument('--manifest',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('plot-confusion',help='600-dpi confusion matrix from classification predictions')
    a.add_argument('--predictions',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('plot-quality',help='600-dpi class confidence/realism/diversity/quality histograms')
    a.add_argument('--augmentation-dir',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('smoke',help='Run completely synthetic non-clinical integration test')
    a.add_argument('--workdir',default='outputs/smoke_run')
    return p

def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=='smoke':
        from .smoke import run_smoke
        result=run_smoke(args.workdir)
        print(json.dumps(result,indent=2));return 0
    cfg=load_config(args.config,args.set)
    if args.command in ('prepare-tmed','prepare-echonet','prepare-camus','audit'):
        from .data import prepare_tmed,prepare_echonet,prepare_camus,load_manifest,audit_splits
        if args.command=='prepare-tmed':
            result={'manifest':prepare_tmed(args.metadata,args.out,args.root,seed=cfg['project']['seed'],
                   max_frames_per_view_study=cfg['data'].get('tmed_max_frames_per_view_study',2),allow_study_only=args.allow_study_only)}
        elif args.command=='prepare-echonet':
            result={'manifest':prepare_echonet(args.root,args.out,args.images,max_videos=args.max_videos,
                    frames_per_video=cfg['data'].get('max_frames_per_video',5),skip_invalid=args.skip_invalid)}
        elif args.command=='prepare-camus':result={'manifest':prepare_camus(args.root,args.out,args.images,seed=cfg['project']['seed'])}
        else:result=audit_splits(load_manifest(args.manifest,args.task))
    elif args.command in ('train-structure','train-probe','train-evaluator','train-teacher','train-gan','train-classifier','eval-classifier'):
        from .train import fit_structure,fit_probe,fit_evaluator,fit_classifier,fit_gan,evaluate_classifier
        if args.command=='train-structure':result=fit_structure(cfg,args.camus,args.echonet,args.out)
        elif args.command=='train-probe':result=fit_probe(cfg,args.camus,args.out)
        elif args.command=='train-evaluator':result=fit_evaluator(cfg,args.manifest,args.out,unlabeled_manifest=args.unlabeled_manifest)
        elif args.command=='train-teacher':result=fit_classifier(cfg,args.manifest,args.out,args.task,teacher=True)
        elif args.command=='train-gan':result=fit_gan(cfg,args.manifest,args.structure,args.evaluator,args.out,args.resume,args.task)
        elif args.command=='train-classifier':result=fit_classifier(cfg,args.manifest,args.out,args.task,args.arch,seed=args.seed)
        else:result=evaluate_classifier(cfg,args.manifest,args.checkpoint,args.split,args.out)
    elif args.command=='train-baseline':
        from .baseline_train import fit_baseline
        result=fit_baseline(cfg,args.manifest,args.name,args.out,args.evaluator,args.task,args.group_class,args.group_view)
    elif args.command in ('augment','train-critic','sample-candidates'):
        from .quality import fit_balanced_manifest,fit_realism_critic,sample_candidate_pool
        if args.command=='train-critic':result=fit_realism_critic(cfg,args.real,args.candidates,args.out,args.task)
        elif args.command=='sample-candidates':
            mapping=json.load(open(args.checkpoint_map,encoding='utf-8')) if args.checkpoint_map else None
            result=sample_candidate_pool(cfg,args.manifest,args.generator,args.out,args.task,args.kind,
                 args.n_per_group,checkpoint_map=mapping,seed=args.seed,sampling_steps=args.sampling_steps)
        else:
            mapping=json.load(open(args.checkpoint_map,encoding='utf-8')) if args.checkpoint_map else None
            result=fit_balanced_manifest(cfg,args.manifest,args.generator,args.teacher,args.structure,args.out,
                task=args.task,kind=args.kind,critic_checkpoint=args.critic,without_gating=args.without_gating,
                checkpoint_map=mapping,seed=args.seed,sampling_steps=args.sampling_steps)
    elif args.command in ('eval-generator','eval-anatomy','eval-external-ef'):
        from .evaluate import evaluate_generator,evaluate_anatomy,evaluate_external_ef
        if args.command=='eval-generator':result=evaluate_generator(cfg,args.manifest,args.checkpoint,args.evaluator,
                                                args.out,args.task,args.kind,args.split,args.max_images,
                                                include_inception=args.inception,radimagenet_weights=args.radimagenet_weights)
        elif args.command=='eval-anatomy':result=evaluate_anatomy(cfg,args.camus,args.probe,args.structure,args.generator,args.out,args.n_generated)
        else:result=evaluate_external_ef(cfg,args.camus,args.checkpoint,args.out)
    elif args.command in ('compare-seeds','bootstrap'):
        from .statistics import compare_seeds,paired_study_bootstrap
        from .utils import write_json
        if args.command=='compare-seeds':result=compare_seeds(args.csv,args.out)
        else:
            result=paired_study_bootstrap(args.a,args.b,args.metric,args.reps);write_json(args.out,result)
    elif args.command=='profile':
        from .profiling import profile_inference
        result=profile_inference(cfg,args.checkpoint,args.out,args.kind,args.batch_size,args.warmup,args.runs,args.sampling_steps)
    elif args.command in ('plot-curves','plot-grid','plot-balance','plot-confusion','plot-quality'):
        from .visualize import training_curves,candidate_grid,class_balance_plot,confusion_matrix_plot,quality_score_plot
        if args.command=='plot-curves':training_curves(args.log,args.out)
        elif args.command=='plot-grid':candidate_grid(args.manifest,args.out)
        elif args.command=='plot-balance':class_balance_plot(args.manifest,args.out)
        elif args.command=='plot-confusion':confusion_matrix_plot(args.predictions,args.out)
        else:quality_score_plot(args.augmentation_dir,args.out)
        result={'saved':args.out}
    else:raise RuntimeError(f'Unexpected command {args.command}')
    print(json.dumps(result,indent=2,default=str))
    return 0

if __name__=='__main__':main()
