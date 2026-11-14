"""Independent auxiliary encoders, segmentation probe and task classifiers."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class TinyEncoder(nn.Module):
    def __init__(self,embed_dim=32):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),nn.MaxPool2d(2))
        self.pool=nn.AdaptiveAvgPool2d(1);self.embed=nn.Linear(32,embed_dim)
        self.seg=nn.Conv2d(32,4,1);self.ef=nn.Linear(embed_dim,2)
    def forward(self,x,return_heads=False):
        feat=self.net(x);emb=self.embed(self.pool(feat).flatten(1))
        if return_heads:
            seg=F.interpolate(self.seg(feat),size=x.shape[-2:],mode='bilinear',align_corners=False)
            return emb,seg,self.ef(emb)
        return emb

class StructureEncoder(nn.Module):
    """ResNet-18 feature encoder with anatomy mask and EF supervision heads.
    Only embedding is used for GAN regularization and quality gate.
    """
    def __init__(self,embed_dim=128,pretrained=False):
        super().__init__()
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        net=models.resnet18(weights=weights)
        self.gray_conv=nn.Conv2d(1,64,7,2,3,bias=False)
        with torch.no_grad():self.gray_conv.weight.copy_(net.conv1.weight.mean(1,keepdim=True))
        self.bn=net.bn1;self.relu=net.relu;self.maxpool=net.maxpool
        self.layer1=net.layer1;self.layer2=net.layer2;self.layer3=net.layer3;self.layer4=net.layer4
        self.embed=nn.Linear(512,embed_dim)
        self.seg=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,4,1))
        self.ef=nn.Linear(embed_dim,2)
    def forward(self,x,return_heads=False):
        h=self.maxpool(self.relu(self.bn(self.gray_conv(x))))
        f1=self.layer1(h);h=self.layer2(f1);h=self.layer3(h);h=self.layer4(h)
        emb=self.embed(F.adaptive_avg_pool2d(h,1).flatten(1))
        if return_heads:
            seg=F.interpolate(self.seg(f1),size=x.shape[-2:],mode='bilinear',align_corners=False)
            return emb,seg,self.ef(emb)
        return emb

class _ConvPair(nn.Module):
    def __init__(self,ci,co):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.GroupNorm(min(8,co),co),nn.ReLU(inplace=True),
                                                  nn.Conv2d(co,co,3,padding=1),nn.GroupNorm(min(8,co),co),nn.ReLU(inplace=True))
    def forward(self,x):return self.net(x)

class UNetProbe(nn.Module):
    def __init__(self,base=24,n_classes=4):
        super().__init__();self.d1=_ConvPair(1,base);self.d2=_ConvPair(base,base*2);self.d3=_ConvPair(base*2,base*4)
        self.b=_ConvPair(base*4,base*8)
        self.u3=_ConvPair(base*12,base*4);self.u2=_ConvPair(base*6,base*2);self.u1=_ConvPair(base*3,base)
        self.out=nn.Conv2d(base,n_classes,1)
    def forward(self,x):
        a=self.d1(x);b=self.d2(F.max_pool2d(a,2));c=self.d3(F.max_pool2d(b,2));d=self.b(F.max_pool2d(c,2))
        u=self.u3(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        u=self.u2(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.out(self.u1(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))

class TinyClassifier(nn.Module):
    def __init__(self,n_classes=3):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,n_classes))
    def forward(self,x):return self.net(x)

class ImageNetClassifier(nn.Module):
    def __init__(self,architecture='efficientnet_b0',n_classes=3,pretrained=True):
        super().__init__();self.architecture=architecture
        if architecture=='efficientnet_b0':
            self.net=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.net.classifier[-1]=nn.Linear(self.net.classifier[-1].in_features,n_classes)
        elif architecture=='densenet121':
            self.net=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            self.net.classifier=nn.Linear(self.net.classifier.in_features,n_classes)
        elif architecture=='resnet18':
            self.net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.net.fc=nn.Linear(self.net.fc.in_features,n_classes)
        elif architecture=='resnet50':
            self.net=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.net.fc=nn.Linear(self.net.fc.in_features,n_classes)
        elif architecture=='vit_b_16':
            self.net=models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            self.net.heads.head=nn.Linear(self.net.heads.head.in_features,n_classes)
        else:raise ValueError(f'Unknown classifier {architecture}')
    def forward(self,x):
        if x.size(1)==1:x=x.repeat(1,3,1,1)
        if self.architecture=='vit_b_16':x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
        return self.net(x)

def create_classifier(architecture,n_classes=3,pretrained=False):
    if architecture=='tiny':return TinyClassifier(n_classes)
    return ImageNetClassifier(architecture,n_classes,pretrained)

def create_structure(cfg):
    if cfg['model']['base_channels']<16:return TinyEncoder(cfg['auxiliary'].get('encoder_dim',32))
    return StructureEncoder(cfg['auxiliary'].get('encoder_dim',128),cfg['model'].get('pretrained',False))
