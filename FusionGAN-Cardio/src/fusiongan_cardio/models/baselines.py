"""Self-contained operational GAN comparator architectures.

stylegan2ada_lite and vitgan_lite are *research reference implementations*
(not bit-for-bit recreations of the original authors' StyleGAN2-ADA / ViTGAN).
Their reported scores MUST use these explicit names, not the canonical papers'.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .core import GlobalDiscriminator

class ConvGenerator(nn.Module):
    def __init__(self,image_size=256,latent_dim=128,classes=3,views=5,base=32,conditional=True):
        super().__init__();self.latent_dim=latent_dim;self.image_size=image_size;self.conditional=conditional
        self.y_emb=nn.Embedding(classes,16) if conditional else None
        self.v_emb=nn.Embedding(views,16) if conditional else None
        zdim=latent_dim+(32 if conditional else 0)
        n=int(math.log2(image_size))-2
        channels=[min(base*8,256)]+[max(base//2,min(base*8,256)//(2**(i+1))) for i in range(n-1)]
        self.fc=nn.Linear(zdim,channels[0]*4*4)
        seq=[]
        for i in range(len(channels)-1):
            seq += [nn.ConvTranspose2d(channels[i],channels[i+1],4,2,1),nn.BatchNorm2d(channels[i+1]),nn.ReLU()]
        seq += [nn.ConvTranspose2d(channels[-1],1,4,2,1),nn.Sigmoid()]
        self.net=nn.Sequential(*seq);self.start=channels[0]
    def forward(self,z,y,v):
        if self.conditional:z=torch.cat([z,self.y_emb(y),self.v_emb(v)],dim=1)
        return self.net(self.fc(z).reshape(len(z),self.start,4,4))

class CondConvCritic(nn.Module):
    def __init__(self,image_size=256,classes=3,views=5,base=32,conditional=True,spectral=False):
        super().__init__();self.conditional=conditional
        self.base=GlobalDiscriminator(classes,views,base,image_size,spectral=spectral)
    def forward(self,x,y,v):
        score,cl,vl,feat=self.base(x,y,v)
        if not self.conditional:
            # Remove projection term for unconditional baselines. Reuse the feature path.
            hidden=F.adaptive_avg_pool2d(feat[-1],1).flatten(1)
            score=self.base.linear(hidden).squeeze(1)
        return {'score':score,'global_score':score,'class_logits':cl,'view_logits':vl,'features':feat}

class AdaIN(nn.Module):
    def __init__(self,channels,style_dim=128):
        super().__init__();self.affine=nn.Linear(style_dim,2*channels)
        nn.init.zeros_(self.affine.weight);nn.init.zeros_(self.affine.bias)
    def forward(self,x,w):
        norm=F.instance_norm(x)
        a,b=self.affine(w).chunk(2,1)
        return norm*(1+a[...,None,None])+b[...,None,None]

class StyleGAN2ADALite(nn.Module):
    """Practical style-modulated generator and per-feature noise, not original StyleGAN2 code."""
    def __init__(self,image_size=256,latent_dim=128,classes=3,views=5,base=32):
        super().__init__();self.latent_dim=latent_dim;self.image_size=image_size
        self.y_emb=nn.Embedding(classes,32);self.v_emb=nn.Embedding(views,32)
        self.mapping=nn.Sequential(nn.Linear(latent_dim+64,128),nn.LeakyReLU(.2),nn.Linear(128,128),nn.LeakyReLU(.2))
        n=int(math.log2(image_size))-2
        channels=[min(8*base,256)]+[max(8, min(base*8,256)//(2**i)) for i in range(1,n+1)]
        self.const=nn.Parameter(torch.randn(1,channels[0],4,4)*.1)
        self.conv=nn.ModuleList([nn.Conv2d(a,b,3,padding=1) for a,b in zip(channels[:-1],channels[1:])])
        self.adain=nn.ModuleList([AdaIN(c) for c in channels[1:]])
        self.noise_gain=nn.ParameterList(nn.Parameter(torch.tensor(0.01)) for _ in self.conv)
        self.to_image=nn.Conv2d(channels[-1],1,1)
    def forward(self,z,y,v):
        w=self.mapping(torch.cat([z,self.y_emb(y),self.v_emb(v)],1));x=self.const.expand(len(z),-1,-1,-1)
        for conv,mod,ng in zip(self.conv,self.adain,self.noise_gain):
            x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
            x=F.leaky_relu(mod(conv(x)+ng*torch.randn((len(x),1,x.size(2),x.size(3)),device=x.device),w),.2)
        return torch.sigmoid(self.to_image(x))

class ViTGANLite(nn.Module):
    """Transformer-token-conditioned GAN synthesis with convolutional progressive decoding."""
    def __init__(self,image_size=256,latent_dim=128,classes=3,views=5,base=32):
        super().__init__();self.latent_dim=latent_dim
        self.y_emb=nn.Embedding(classes,64);self.v_emb=nn.Embedding(views,64)
        self.token_proj=nn.Linear(latent_dim+128,16*128)
        self.pos=nn.Parameter(torch.randn(1,16,128)*.02)
        layer=nn.TransformerEncoderLayer(d_model=128,nhead=4,dim_feedforward=256,dropout=.0,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.decoder=ConvGenerator(image_size,latent_dim,classes,views,base,conditional=True)
        # Transformer tokens projected to latent decoder input.
        self.to_latent=nn.Linear(16*128,latent_dim)
    def forward(self,z,y,v):
        cond=torch.cat([z,self.y_emb(y),self.v_emb(v)],dim=1)
        t=self.transformer(self.token_proj(cond).reshape(len(z),16,128)+self.pos)
        updated=self.to_latent(t.reshape(len(z),-1))
        return self.decoder(updated,y,v)

def build_baseline(name,cfg,classes=3,views=5):
    m=cfg['model'];size=cfg['data']['image_size'];zdim=m['latent_dim'];base=m['discriminator_base']
    if name in ('dcgan','wgangp'):
        g=ConvGenerator(size,zdim,classes,views,base,conditional=False)
        d=CondConvCritic(size,classes,views,base,conditional=False,spectral=False)
    elif name=='cgan':
        g=ConvGenerator(size,zdim,classes,views,base,conditional=True)
        d=CondConvCritic(size,classes,views,base,conditional=True,spectral=True)
    elif name=='stylegan2ada_lite':
        g=StyleGAN2ADALite(size,zdim,classes,views,base)
        d=CondConvCritic(size,classes,views,base,conditional=True,spectral=True)
    elif name=='vitgan_lite':
        g=ViTGANLite(size,zdim,classes,views,base)
        d=CondConvCritic(size,classes,views,base,conditional=True,spectral=True)
    else:raise ValueError(f'Unknown GAN baseline {name}')
    return g,d
