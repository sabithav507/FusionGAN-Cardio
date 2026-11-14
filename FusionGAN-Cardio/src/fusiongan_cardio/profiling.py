"""Measured (never fabricated) inference runtime and memory profiling.
Assumes preprocessing and data-transfer excluded; batch/device described in output.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import torch
from .utils import get_device,count_params,write_json
from .train import load_gan,load_classifier
from .baseline_train import load_baseline

def profile_inference(cfg,checkpoint,out,kind='fusion',batch_size=1,warmup=50,runs=1000,sampling_steps=50):
    if batch_size<1 or runs<1 or warmup<0:raise ValueError('Invalid profiling batch/runs/warmup')
    device=get_device(cfg['project']['device'])
    if kind=='fusion':net,_,info=load_gan(checkpoint,device)
    elif kind=='baseline':net,info=load_baseline(checkpoint,device)
    elif kind=='classifier':net,info=load_classifier(checkpoint,device)
    else:raise ValueError(kind)
    image_size=info.get('image_size',cfg['data']['image_size'])
    x=torch.rand(batch_size,1,image_size,image_size,device=device)
    y=torch.zeros(batch_size,device=device,dtype=torch.long)
    v=torch.full((batch_size,),2,device=device,dtype=torch.long)
    z=torch.randn(batch_size,info.get('model_config',cfg['model'])['latent_dim'],device=device)
    def infer():
        if kind=='classifier':return net(x)
        if hasattr(net,'sample'):return net.sample(batch_size,y,v,steps=sampling_steps)
        return net(z,y,v)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):infer()
        sync();times=[]
        for _ in range(runs):
            start=time.perf_counter();infer();sync()
            times.append((time.perf_counter()-start)*1000)
    arr=np.asarray(times)
    report={'model':kind,'checkpoint':str(checkpoint),'device':str(device),'batch_size':batch_size,
            'image_size':image_size,'warmup_iterations':warmup,'timed_iterations':runs,
            'latency_mean_ms_per_batch':float(arr.mean()),'latency_sd_ms_per_batch':float(arr.std(ddof=1)) if runs>1 else 0,
            'latency_p95_ms_per_batch':float(np.percentile(arr,95)),
            'latency_mean_ms_per_image':float(arr.mean()/batch_size),
            'throughput_images_per_sec':float(batch_size*1000/arr.mean()),
            'trainable_parameters':count_params(net),
            'peak_cuda_allocated_mb':float(torch.cuda.max_memory_allocated(device)/1048576) if device.type=='cuda' else None,
            'warning':'Inference only; excludes file I/O, image preprocessing, CPU-to-GPU transfer and training time.'}
    write_json(out,report);return report
