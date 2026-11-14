"""Independent SimCLR-like feature extractor for medical-domain Fréchet evaluation.
Never use this encoder for GAN regularization or synthetic sample selection.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class SimCLREncoder(nn.Module):
    def __init__(self,projection_dim=128,small=False):
        super().__init__();self.small=small
        if small:
            self.backbone=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                  nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
            feat_dim=32
        else:
            network=models.resnet50(weights=None)
            network.conv1=nn.Conv2d(1,64,7,stride=2,padding=3,bias=False)
            self.backbone=nn.Sequential(*list(network.children())[:-1],nn.Flatten());feat_dim=2048
        self.projection=nn.Sequential(nn.Linear(feat_dim,max(128,projection_dim)),nn.ReLU(),nn.Linear(max(128,projection_dim),projection_dim))
    def forward(self,x,project=False):
        features=self.backbone(x)
        return F.normalize(self.projection(features),dim=1) if project else features

def contrastive_loss(u,v,temperature=.2):
    if len(u)!=len(v):raise ValueError('Unequal SimCLR views')
    b=len(u)
    if b<2:return (u.sum()+v.sum())*0
    f=F.normalize(torch.cat([u,v],dim=0),dim=1)
    sim=(f@f.T)/temperature
    sim.fill_diagonal_(-1e9)
    labels=(torch.arange(2*b,device=f.device)+b)%(2*b)
    return F.cross_entropy(sim,labels)

def augment_ultrasound(x):
    # No horizontal flip; intensity/weak multiplicative-speckle/noise changes only.
    contrast=torch.empty(x.shape[0],1,1,1,device=x.device).uniform_(.9,1.1)
    brightness=torch.empty(x.shape[0],1,1,1,device=x.device).uniform_(-.06,.06)
    y=(x*contrast+brightness).clamp(0,1)
    return (y+y*torch.randn_like(y)*.03).clamp(0,1)

class IndependentFeatureEncoder(nn.Module):
    """Held-out alternative evaluators: ImageNet Inception-v3 and user-supplied RadImageNet ResNet-50.

    These models are never shared with structural regularization, teachers, or gates.
    A random-weight checkpoint is rejected; Inception requires ImageNet weights.
    """
    def __init__(self,kind,weights_file=None):
        super().__init__();self.kind=kind
        if kind=='inception_v3':
            net=models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
            net.fc=nn.Identity();self.size=299
        elif kind=='radimagenet_resnet50':
            if not weights_file:raise ValueError('RadImageNet evaluation needs explicit pretrained --radimagenet-weights')
            from pathlib import Path
            if not Path(weights_file).is_file():raise FileNotFoundError(weights_file)
            net=models.resnet50(weights=None)
            try:ck=torch.load(weights_file,map_location='cpu',weights_only=False)
            except TypeError:ck=torch.load(weights_file,map_location='cpu')
            if not isinstance(ck,dict):raise ValueError('Unsupported RadImageNet checkpoint layout')
            state=ck.get('state_dict',ck.get('model_state_dict',ck.get('model',ck)))
            state={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in state.items() if torch.is_tensor(v)}
            state={k:v for k,v in state.items() if not k.startswith('fc.')}
            missing,unexpected=net.load_state_dict(state,strict=False)
            if len([k for k in missing if not k.startswith('fc.')])>0:
                raise ValueError(f'RadImageNet encoder weights incomplete: {missing[:10]}')
            net.fc=nn.Identity();self.size=224
        else:raise ValueError(kind)
        self.net=net
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
    def forward(self,x):
        if x.size(1)==1:x=x.repeat(1,3,1,1)
        x=F.interpolate(x,size=(self.size,self.size),mode='bilinear',align_corners=False)
        return self.net((x-self.mean)/self.std)
