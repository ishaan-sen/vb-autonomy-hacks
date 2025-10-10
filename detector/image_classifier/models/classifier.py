import torch
import torch.nn as nn
import torchvision.models as tv
import timm

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "convnext_tiny":
        m = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_feats, num_classes)
        return m
    elif name == "vit_b_16":
        m = tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1)
        in_feats = m.heads.head.in_features
        m.heads.head = nn.Linear(in_feats, num_classes)
        return m
    elif name.startswith("timm:"):
        # example: timm:eva02_large_patch14_448.mim_in22k_ft_in1k
        timm_name = name.split("timm:")[1]
        m = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
        return m
    else:
        raise ValueError(f"Unknown model {name}")
