"""FusionGAN-Cardio: conditional residual generator with FiLM, gated fusion and CBAM;
global projection + 70x70 receptive-field conditional PatchGAN discriminator.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

class ConditionalResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,cond_dim,upsample=True):
        super().__init__();self.upsample=upsample
        self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.bn1=nn.BatchNorm2d(out_ch);self.bn2=nn.BatchNorm2d(out_ch)
        self.skip=nn.Conv2d(in_ch,out_ch,1)
        self.film=nn.Linear(cond_dim,2*out_ch)
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
    def forward(self,x,condition):
        up=F.interpolate(x,scale_factor=2,mode='nearest') if self.upsample else x
        z=self.bn1(self.conv1(up))
        scale,shift=self.film(condition).chunk(2,dim=1)
        z=z*(1+scale[...,None,None])+shift[...,None,None]
        z=F.relu(z,inplace=False)
        z=self.bn2(self.conv2(z))
        return F.relu((z+self.skip(up))/math.sqrt(2),inplace=False)

class CBAM(nn.Module):
    def __init__(self,channels,reduction=8):
        super().__init__();inter=max(2,channels//reduction)
        self.mlp=nn.Sequential(nn.Conv2d(channels,inter,1,bias=False),nn.ReLU(),nn.Conv2d(inter,channels,1,bias=False))
        self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x,return_maps=False):
        channel=torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x,1))+self.mlp(F.adaptive_max_pool2d(x,1)))
        ch=x*channel
        spatial=torch.sigmoid(self.spatial(torch.cat((ch.mean(dim=1,keepdim=True),ch.amax(dim=1,keepdim=True)),dim=1)))
        refined=ch*spatial
        return (refined,channel,spatial) if return_maps else refined

class FusionGenerator(nn.Module):
    def __init__(self,num_classes=3,n_views=5,image_size=256,latent_dim=128,embed_dim=32,base_channels=64,
                 fused_channels=64,attention_reduction=8,adaptive_fusion=True,dual_attention=True):
        super().__init__()
        if image_size<32 or (image_size & (image_size-1)):raise ValueError('image_size must be >=32 and power of 2')
        self.latent_dim=latent_dim;self.image_size=image_size
        self.y_emb=nn.Embedding(num_classes,embed_dim);self.v_emb=nn.Embedding(n_views,embed_dim)
        cond_dim=embed_dim*2
        base=base_channels;self.initial_channels=base*8
        self.proj=nn.Linear(latent_dim+cond_dim,self.initial_channels*4*4)
        nblocks=int(math.log2(image_size))-3  # 4x4 -> size/2. Full: 5 blocks to 128.
        channel_plan=[base*8,base*4,base*4,base*2,base]
        if nblocks>len(channel_plan):channel_plan.extend([base]* (nblocks-len(channel_plan)))
        channel_plan=channel_plan[:nblocks]
        self.blocks=nn.ModuleList()
        previous=self.initial_channels
        for channels in channel_plan:
            self.blocks.append(ConditionalResBlock(previous,channels,cond_dim,upsample=True));previous=channels
        # For reduced smoke size, include initial representation to ensure three taps.
        tap_channels=([self.initial_channels]+channel_plan)[-3:]
        self.align=nn.ModuleList(nn.Conv2d(ch,fused_channels,1) for ch in tap_channels)
        self.adaptive_fusion=adaptive_fusion;self.dual_attention=dual_attention
        self.gate=nn.Sequential(nn.Linear(3*fused_channels+cond_dim,max(32,fused_channels)),nn.ReLU(),nn.Linear(max(32,fused_channels),3))
        self.cbam=CBAM(fused_channels,attention_reduction)
        self.preliminary=nn.Conv2d(previous,1,3,padding=1)
        self.final=ConditionalResBlock(fused_channels,max(4,base//2),cond_dim,upsample=True)
        self.output_conv=nn.Conv2d(max(4,base//2),1,3,padding=1)
    def forward(self,z,y,view,return_aux=False):
        if z.ndim!=2 or z.shape[1]!=self.latent_dim:raise ValueError(f'Expected z [B,{self.latent_dim}]')
        if y.shape!=view.shape or y.ndim!=1 or y.shape[0]!=z.shape[0]:raise ValueError('Condition batch shapes mismatch')
        condition=torch.cat([self.y_emb(y),self.v_emb(view)],dim=1)
        h=self.proj(torch.cat([z,condition],dim=1)).reshape(z.size(0),self.initial_channels,4,4)
        stages=[h]
        for block in self.blocks:
            h=block(h,condition);stages.append(h)
        taps=stages[-3:]
        target=(self.image_size//2,self.image_size//2)
        aligned=[F.interpolate(proj(t),size=target,mode='bilinear',align_corners=False)
                 for t,proj in zip(taps,self.align)]
        if self.adaptive_fusion:
            pooled=torch.cat([F.adaptive_avg_pool2d(t,1).flatten(1) for t in aligned]+[condition],dim=1)
            weights=torch.softmax(self.gate(pooled),dim=1)
        else:weights=z.new_full((z.shape[0],3),1.0/3)
        fused=sum(w[:,None,None,None]*level for w,level in zip(weights.unbind(1),aligned))
        refined,channel_map,spatial_map=(self.cbam(fused,return_maps=True) if self.dual_attention else
                                          (fused,None,None))
        out=self.final(refined,condition)
        image=torch.sigmoid(self.output_conv(out))
        if return_aux:
            return {'image':image,'fusion_weights':weights,'preliminary':torch.sigmoid(self.preliminary(h)),
                    'channel_attention':channel_map,'spatial_attention':spatial_map}
        return image

class _DiscConv(nn.Module):
    def __init__(self,ci,co,stride=2,spectral=True):
        super().__init__();layer=nn.Conv2d(ci,co,4,stride=stride,padding=1)
        self.conv=spectral_norm(layer) if spectral else layer
    def forward(self,x):return F.leaky_relu(self.conv(x),negative_slope=.2,inplace=False)

class GlobalDiscriminator(nn.Module):
    def __init__(self,num_classes=3,n_views=5,base=32,image_size=256,spectral=True):
        super().__init__()
        steps=max(3,int(math.log2(image_size))-3)
        chans=[base,min(base*2,256),min(base*4,256),min(base*8,512),min(base*8,512),min(base*8,512)]
        if steps>len(chans):chans.extend([chans[-1]]*(steps-len(chans)))
        layers=[];prior=1
        for ch in chans[:steps]:layers.append(_DiscConv(prior,ch,2,spectral));prior=ch
        self.layers=nn.ModuleList(layers)
        wrap=(lambda x:spectral_norm(x)) if spectral else (lambda x:x)
        self.linear=wrap(nn.Linear(prior,1));self.y_emb=nn.Embedding(num_classes,prior);self.v_emb=nn.Embedding(n_views,prior)
        self.class_head=wrap(nn.Linear(prior,num_classes));self.view_head=wrap(nn.Linear(prior,n_views))
    def forward(self,x,y,view):
        features=[]
        for layer in self.layers:
            x=layer(x);features.append(x)
        f=F.adaptive_avg_pool2d(x,1).flatten(1)
        # Projection conditional score (Miyato and Koyama style).
        score=self.linear(f).squeeze(1)+((self.y_emb(y)+self.v_emb(view))*f).sum(dim=1)/math.sqrt(f.shape[1])
        return score,self.class_head(f),self.view_head(f),features

class PatchDiscriminator(nn.Module):
    """k4, strides [2,2,2,1,1], p1 => effective receptive field 70.
    For 256x256, output is exactly 30x30.
    """
    def __init__(self,num_classes=3,n_views=5,base=32,spectral=True):
        super().__init__();channels=[base,base*2,base*4,base*8,base*8]
        self.layers=nn.ModuleList();ci=1
        for ch,s in zip(channels,[2,2,2,1,1]):
            self.layers.append(_DiscConv(ci,ch,s,spectral));ci=ch
        self.score=spectral_norm(nn.Conv2d(ci,1,1)) if spectral else nn.Conv2d(ci,1,1)
        self.y_emb=nn.Embedding(num_classes,ci);self.v_emb=nn.Embedding(n_views,ci)
    def forward(self,x,y,view):
        features=[]
        # Input must be >= 32 for the 5-layer patch network to have positive extent.
        for layer in self.layers:
            x=layer(x);features.append(x)
        cond=(self.y_emb(y)+self.v_emb(view))[:, :, None, None]
        scores=self.score(x)+(cond*x).sum(1,keepdim=True)/math.sqrt(x.shape[1])
        return scores,features

class MultiScaleDiscriminator(nn.Module):
    def __init__(self,num_classes=3,n_views=5,base=32,image_size=256,multi_scale=True):
        super().__init__();self.multi_scale=multi_scale
        self.global_branch=GlobalDiscriminator(num_classes,n_views,base,image_size)
        self.patch_branch=PatchDiscriminator(num_classes,n_views,base)
    def forward(self,x,y,view):
        score,cl,vl,global_feats=self.global_branch(x,y,view)
        patch,patch_feats=self.patch_branch(x,y,view) if self.multi_scale else (None,[])
        s=.5*score+.5*patch.flatten(1).mean(1) if patch is not None else score
        feats=(global_feats[-2:]+patch_feats[-2:]) if patch is not None else global_feats[-4:]
        return {'score':s,'global_score':score,'patch_map':patch,'class_logits':cl,'view_logits':vl,'features':feats}


def make_gan(cfg,num_classes=3,n_views=5):
    m=cfg['model'];a=m.get('ablation',{})
    gen=FusionGenerator(num_classes,n_views,cfg['data']['image_size'],m['latent_dim'],m['condition_embed_dim'],
                        m['base_channels'],m['fused_channels'],m.get('attention_reduction',8),
                        a.get('adaptive_fusion',True),a.get('dual_attention',True))
    disc=MultiScaleDiscriminator(num_classes,n_views,m['discriminator_base'],cfg['data']['image_size'],a.get('multi_scale',True))
    return gen,disc
