import torch
import torch.nn as nn
import timm


class VisionTransformer(nn.Module):
    def __init__(self, model_type, pretrained=False, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_type, pretrained=pretrained, **kwargs)

    def forward(self, x):
        return self.model(x)


def vit_small(pretrained=False, **kwargs):
    """ViT-Small for CIFAR
    - img_size=32 (CIFAR image size)
    - patch_size=4
    - embed_dim=384
    - depth=12
    - num_heads=6
    """
    return VisionTransformer(
        model_type="vit_base_patch16_224",
        pretrained=pretrained,
        img_size=32,
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )


def vit_tiny(pretrained=False, **kwargs):
    """ViT-Tiny for CIFAR
    - img_size=32 (CIFAR image size)
    - patch_size=4
    - embed_dim=192
    - depth=12
    - num_heads=3
    """
    return VisionTransformer(
        model_type="vit_base_patch16_224",
        pretrained=pretrained,
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs
    )


def vit_base(pretrained=False, **kwargs):
    """ViT-Base for CIFAR
    - img_size=32 (CIFAR image size)
    - patch_size=4
    - embed_dim=768
    - depth=12
    - num_heads=12
    """
    return VisionTransformer(
        model_type="vit_base_patch16_224",
        pretrained=pretrained,
        img_size=32,
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )


# def vit_b_32(pretrained=False, **kwargs):
#     """ViT-B/32 for CIFAR
#     - img_size=32 (CIFAR image size)
#     - patch_size=4
#     - embed_dim=768
#     - depth=12
#     - num_heads=12
#     """
#     return VisionTransformer(
#         model_type="vit_base_patch32_224",
#         pretrained=pretrained,
#         img_size=224,
#         patch_size=4,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         **kwargs
#     )

from torchvision.models import vit_b_32, ViT_B_32_Weights


def vit_b_32_(pretrained=False, num_classes=100, **kwargs):
    weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_32(weights=weights, **kwargs)
    if num_classes != 1000:
        model.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=num_classes)
        )
    return model
