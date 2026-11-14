import numpy as np
import torch
import pytest
from fusiongan_cardio.config import load_config
from fusiongan_cardio.models import make_gan
from fusiongan_cardio.utils import count_params
from fusiongan_cardio.losses import (hinge_discriminator,feature_matching,ReferenceBank,mode_seeking,total_variation)

def cfg():return load_config('configs/smoke.yaml')

def test_core_tensor_shapes_and_fusion_weights():
    torch.set_num_threads(1)
    gen,disc=make_gan(cfg())
    z=torch.randn(2,16);y=torch.tensor([0,1]);v=torch.tensor([2,3])
    aux=gen(z,y,v,return_aux=True)
    assert aux['image'].shape==(2,1,32,32)
    assert aux['preliminary'].shape==(2,1,16,16)
    assert torch.all(aux['fusion_weights']>=0)
    assert torch.allclose(aux['fusion_weights'].sum(1),torch.ones(2),atol=1e-6)
    assert aux['spatial_attention'].shape==(2,1,16,16)
    result=disc(aux['image'],y,v)
    assert result['score'].shape==(2,)
    assert result['patch_map'].shape==(2,1,2,2)
    assert result['class_logits'].shape==(2,3)
    assert result['view_logits'].shape==(2,5)
    assert len(result['features'])==4
    assert count_params(gen)>0


def test_full_size_patch_receptive_output():
    from fusiongan_cardio.models.core import PatchDiscriminator
    m=PatchDiscriminator(base=8)
    with torch.no_grad():x=m(torch.rand(1,1,256,256),torch.tensor([0]),torch.tensor([3]))
    assert x[0].shape==(1,1,30,30)


def test_gradient_passes_through_generator_discriminator():
    g,d=make_gan(cfg())
    z=torch.randn(2,16);y=torch.tensor([0,1]);v=torch.tensor([2,3])
    real=torch.rand(2,1,32,32)
    loss=hinge_discriminator(d(real,y,v),d(g(z,y,v).detach(),y,v))
    loss.backward()
    assert any(p.grad is not None for p in d.parameters())
    d.zero_grad();g.zero_grad()
    out=d(g(z,y,v),y,v)
    pred_real=d(real,y,v)
    generator_loss=-out['score'].mean()+feature_matching(pred_real,out)+mode_seeking(z,torch.randn_like(z),g(z,y,v),g(torch.randn_like(z),y,v))+total_variation(g(z,y,v))
    generator_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in g.parameters())


def test_real_reference_bank_same_condition_only():
    emb=torch.tensor([[0.,0.],[100.,100.],[1.,0.]])
    bank=ReferenceBank(emb,torch.tensor([0,1,0]),torch.tensor([2,2,3]))
    nn,dist=bank.nearest(torch.tensor([[.1,0.]]),torch.tensor([0]),torch.tensor([2]))
    assert torch.allclose(nn[0],emb[0])
    with pytest.raises(ValueError,match='Empty'):bank.nearest(torch.rand(1,2),torch.tensor([2]),torch.tensor([4]))
