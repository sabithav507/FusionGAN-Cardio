import torch
import numpy as np
from fusiongan_cardio.config import load_config
from fusiongan_cardio.models.baselines import build_baseline
from fusiongan_cardio.models.diffusion import ConditionalDiffusion
from fusiongan_cardio.statistics import paired_significance,holm_adjust,mean_sd_ci

def test_baseline_gan_shapes():
    torch.set_num_threads(1);cfg=load_config('configs/smoke.yaml')
    y=torch.tensor([0,1]);v=torch.tensor([2,3]);z=torch.randn(2,16)
    for name in ('dcgan','cgan','wgangp','stylegan2ada_lite','vitgan_lite'):
        gen,disc=build_baseline(name,cfg)
        syn=gen(z,y,v)
        assert syn.shape==(2,1,32,32),name
        assert torch.isfinite(syn).all()
        assert disc(syn,y,v)['score'].shape==(2,)


def test_diffusion_loss_sampling_pixel_and_latent():
    torch.set_num_threads(1)
    for latent in (False,True):
        m=ConditionalDiffusion(32,classes=3,views=5,base=8,latent=latent,timesteps=30)
        y=torch.tensor([0,1]);v=torch.tensor([2,3]);x=torch.rand(2,1,32,32)
        loss=m.loss(x,y,v)
        assert loss.isfinite();loss.backward()
        s=m.sample(2,y,v,steps=3)
        assert s.shape==x.shape
        assert 0<=s.min().item() and s.max().item()<=1


def test_exact_wilcoxon_limits_and_holm():
    x=np.arange(10,dtype=float);y=x+2
    result=paired_significance(x,y)
    assert np.isclose(result['wilcoxon_two_sided_p'],2/1024)
    result5=paired_significance(x[:5],y[:5])
    assert np.isclose(result5['wilcoxon_two_sided_p'],2/32)
    adjusted=holm_adjust([.001,.01,.03])
    assert all(0<=i<=1 for i in adjusted)
    assert adjusted==sorted(adjusted)
    assert mean_sd_ci([1,2,3])['n']==3
