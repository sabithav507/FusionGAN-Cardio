"""Conditional pixel DDPM and latent diffusion, with working DDIM sampling.
Designed as a compact, reproducible reference baseline, not a pretrained
state-of-the-art Stable Diffusion model.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

def sinusoidal_embedding(t,dim):
    half=dim//2;scale=math.log(10000)/(max(1,half-1))
    freqs=torch.exp(torch.arange(half,device=t.device,dtype=torch.float32)*-scale)
    emb=torch.cat((torch.sin(t[:,None].float()*freqs),torch.cos(t[:,None].float()*freqs)),dim=1)
    return F.pad(emb,(0,dim-emb.size(1))) if dim%2 else emb

class DiffusionBlock(nn.Module):
    def __init__(self,ci,co,cond_dim):
        super().__init__();self.c1=nn.Conv2d(ci,co,3,padding=1);self.c2=nn.Conv2d(co,co,3,padding=1)
        groups=lambda c:min([g for g in (8,4,2,1) if c%g==0],default=1)
        self.n1=nn.GroupNorm(groups(co),co);self.n2=nn.GroupNorm(groups(co),co)
        self.cond=nn.Linear(cond_dim,co)
        self.skip=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity()
    def forward(self,x,c):
        h=F.silu(self.n1(self.c1(x))+self.cond(c)[...,None,None])
        h=self.n2(self.c2(h))
        return F.silu(h+self.skip(x))

class ConditionalUNet(nn.Module):
    def __init__(self,input_channels=1,num_classes=3,n_views=5,base=32):
        super().__init__();base=max(8,base)
        cond_dim=base*4
        self.timestep=nn.Sequential(nn.Linear(cond_dim,cond_dim),nn.SiLU(),nn.Linear(cond_dim,cond_dim))
        self.y_emb=nn.Embedding(num_classes,cond_dim);self.v_emb=nn.Embedding(n_views,cond_dim)
        self.start=DiffusionBlock(input_channels,base,cond_dim)
        self.down=DiffusionBlock(base,base*2,cond_dim)
        self.middle=DiffusionBlock(base*2,base*2,cond_dim)
        self.up=DiffusionBlock(base*3,base,cond_dim)
        self.output=nn.Conv2d(base,input_channels,1)
        self.cond_dim=cond_dim
    def forward(self,x,t,y,v):
        c=self.timestep(sinusoidal_embedding(t,self.cond_dim))+self.y_emb(y)+self.v_emb(v)
        a=self.start(x,c);b=self.down(F.avg_pool2d(a,2),c);b=self.middle(b,c)
        h=self.up(torch.cat([F.interpolate(b,size=a.shape[-2:],mode='nearest'),a],dim=1),c)
        return self.output(h)

class LatentAutoencoder(nn.Module):
    """Four-channel 4x-compressed image latent with trainable decoder; KL regularization."""
    def __init__(self,latent_channels=4):
        super().__init__();self.latent_channels=latent_channels
        self.encoder=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),
              nn.Conv2d(64,64,4,2,1),nn.ReLU(),nn.Conv2d(64,2*latent_channels,3,padding=1))
        self.decoder=nn.Sequential(nn.Conv2d(latent_channels,64,3,padding=1),nn.ReLU(),
              nn.ConvTranspose2d(64,64,4,2,1),nn.ReLU(),nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.Conv2d(32,1,3,padding=1))
    def encode(self,x,sample=True):
        mu,logvar=self.encoder(x).chunk(2,dim=1);logvar=logvar.clamp(-20,10)
        return (mu+torch.randn_like(mu)*torch.exp(.5*logvar) if sample else mu),mu,logvar
    def decode(self,z):return torch.sigmoid(self.decoder(z))
    def forward(self,x):
        z,mu,logvar=self.encode(x)
        out=self.decode(z);kl=-.5*(1+logvar-mu.square()-logvar.exp()).mean()
        return out,F.l1_loss(out,x)+.0001*kl

class ConditionalDiffusion(nn.Module):
    def __init__(self,image_size=256,classes=3,views=5,base=32,latent=False,timesteps=1000):
        super().__init__();self.image_size=image_size;self.is_latent=latent;self.timesteps=int(timesteps)
        self.vae=LatentAutoencoder(4) if latent else None
        channels=4 if latent else 1
        self.noise_model=ConditionalUNet(channels,classes,views,base)
        beta=torch.linspace(1e-4,.02,self.timesteps)
        alphas=1-beta;cumprod=torch.cumprod(alphas,dim=0)
        self.register_buffer('betas',beta);self.register_buffer('alpha_bars',cumprod)
    def loss(self,x,y,v):
        if self.vae is not None:
            with torch.no_grad():x,_,_=self.vae.encode(x,sample=False)
        n=len(x);t=torch.randint(0,self.timesteps,(n,),device=x.device)
        e=torch.randn_like(x)
        ap=self.alpha_bars[t].reshape(-1,1,1,1)
        noised=ap.sqrt()*x+(1-ap).sqrt()*e
        estimate=self.noise_model(noised,t,y,v)
        return F.mse_loss(estimate,e)
    @torch.no_grad()
    def sample(self,n,y,v,steps=50,generator=None):
        device=next(self.parameters()).device
        size=self.image_size//4 if self.is_latent else self.image_size
        x=torch.randn((n,4 if self.is_latent else 1,size,size),device=device,generator=generator)
        timeline=torch.linspace(self.timesteps-1,0,min(steps,self.timesteps),device=device).long()
        timeline=torch.unique_consecutive(timeline)
        for i,t in enumerate(timeline):
            t=int(t);vec=torch.full((n,),t,device=device,dtype=torch.long)
            eps=self.noise_model(x,vec,y,v)
            at=self.alpha_bars[t]
            clean=(x-(1-at).sqrt()*eps)/at.sqrt()
            # DDIM eta=0 deterministic transitions; final range mapped to [0,1].
            if i+1 < len(timeline):
                prev=int(timeline[i+1]);a_prev=self.alpha_bars[prev]
                x=a_prev.sqrt()*clean+(1-a_prev).sqrt()*eps
            else:x=clean
        return self.vae.decode(x) if self.vae is not None else x.clamp(0,1)
    def forward(self,z,y,v):
        return self.sample(len(z),y,v,steps=50)
