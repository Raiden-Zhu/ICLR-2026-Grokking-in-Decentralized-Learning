import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


def _should_force_quick_gelu(model_name, pretrained):
    """Enable QuickGELU when loading OpenAI ViT-B-32 pretrained weights."""
    if not isinstance(model_name, str) or model_name != "ViT-B-32":
        return False
    if not isinstance(pretrained, str):
        return False
    return pretrained.strip().lower() == "openai"


def _deterministic_positional_init(num_tokens, embed_dim, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    initialized = torch.randn((num_tokens, embed_dim), generator=generator) * 0.02
    return initialized.to(device=device, dtype=dtype)


def _interpolate_positional_embedding(positional_embedding, target_grid_size):
    """Interpolate CLIP ViT positional embeddings to a new patch grid size."""
    if positional_embedding.ndim != 2 or positional_embedding.shape[0] < 2:
        return None

    num_tokens, embed_dim = positional_embedding.shape
    patch_tokens = num_tokens - 1
    old_grid_size = int(math.sqrt(patch_tokens))
    if old_grid_size * old_grid_size != patch_tokens:
        return None

    cls_token = positional_embedding[:1]
    patch_pos = positional_embedding[1:]
    patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, embed_dim).permute(0, 3, 1, 2)
    resized = F.interpolate(
        patch_pos,
        size=(target_grid_size, target_grid_size),
        mode="bicubic",
        align_corners=False,
    )
    resized = resized.permute(0, 2, 3, 1).reshape(target_grid_size * target_grid_size, embed_dim)
    return torch.cat([cls_token, resized], dim=0)

# class ImageEncoder(torch.nn.Module):
#     def __init__(self, model_name, keep_lang=False, openclip_cachedir="./cache"):
#         super().__init__()
#         self.model, _, _ = open_clip.create_model_and_transforms(
#             model_name, pretrained="openai", cache_dir=openclip_cachedir
#         )
#         if not keep_lang and hasattr(self.model, "transformer"):
#             delattr(self.model, "transformer")
#         # print("Positional embedding shape:", self.model.visual.positional_embedding.shape)

#     def forward(self, images):
#         # print("Images shape before encoding:", images.shape)
#         return self.model.encode_image(images)

class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name,
        keep_lang=False,
        openclip_cachedir="./cache",
        input_size=64,
        pretrained="openai",
    ):
        super().__init__()

        # Determine cache directory priority: explicit arg > env override > HF home
        cache_dir = openclip_cachedir
        env_cache = os.environ.get("OPENCLIP_CACHE_DIR")
        if env_cache:
            cache_dir = env_cache
        elif cache_dir is None:
            cache_dir = os.environ.get("HF_HOME")

        # Resolve pretrained option (bool -> tag/path, env override supported)
        if isinstance(pretrained, bool):
            if pretrained:
                pretrained = os.environ.get("OPENCLIP_PRETRAINED_PATH", "openai")
            else:
                pretrained = None
        elif pretrained in {"", "none", None}:
            pretrained = None
        else:
            expanded = os.path.expanduser(str(pretrained))
            if os.path.isfile(expanded):
                pretrained = expanded

        force_quick_gelu = _should_force_quick_gelu(model_name, pretrained)

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            force_quick_gelu=force_quick_gelu,
            cache_dir=cache_dir,
        )
        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

        # Adapt positional embeddings to the configured input image size.
        patch_size_attr = self.model.visual.patch_size
        patch_size = int(patch_size_attr[0] if isinstance(patch_size_attr, tuple) else patch_size_attr)
        target_grid_size = max(1, int(input_size) // patch_size)
        target_tokens = target_grid_size * target_grid_size + 1

        current_pos = self.model.visual.positional_embedding
        if current_pos.shape[0] != target_tokens:
            resized = _interpolate_positional_embedding(current_pos, target_grid_size)
            if resized is None:
                resized = _deterministic_positional_init(
                    target_tokens,
                    current_pos.shape[-1],
                    current_pos.device,
                    current_pos.dtype,
                )
            else:
                resized = resized.to(device=current_pos.device, dtype=current_pos.dtype)
            self.model.visual.positional_embedding = nn.Parameter(resized)

    def forward(self, images):
        return self.model.encode_image(images)


# class CLIPclassification(nn.Module):
#     def __init__(self, model_name, nb_clases=200, freeze_clip=False, keep_lang=False):
#         super().__init__()
#         self.image_encoder = ImageEncoder(model_name, keep_lang)
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 3, 224, 224)
#             clip_output = self.image_encoder(dummy_input)
#             clip_dim = clip_output.shape[1]
#         self.classifier = nn.Sequential(
#             nn.Linear(clip_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, nb_clases),
#         )
#         if freeze_clip:
#             for param in self.image_encoder.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         features = self.image_encoder(x)
#         # print("Input shape:", x.shape)
#         return self.classifier(features)

#     def unfreeze_clip(self):
#         for param in self.image_encoder.parameters():
#             param.requires_grad = True

class CLIPClassification(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=200,
        freeze_clip=False,
        keep_lang=False,
        input_size=64,
        openclip_cachedir="./cache",
        pretrained="openai",
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            model_name,
            keep_lang,
            openclip_cachedir=openclip_cachedir,
            input_size=input_size,
            pretrained=pretrained,
        )
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            clip_output = self.image_encoder(dummy_input)
            clip_dim = clip_output.shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        if freeze_clip:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.image_encoder(x)
        return self.classifier(features)


CLIPclassification = CLIPClassification

def get_CLIPclassification_model(
    model_name,
    num_classes=200,
    freeze_clip=False,
    keep_lang=False,
    input_size=64,
    openclip_cachedir="./cache",
    pretrained="openai",
    nb_clases=None,
):
    if nb_clases is not None:
        num_classes = nb_clases
    return CLIPClassification(
        model_name,
        num_classes,
        freeze_clip,
        keep_lang,
        input_size=input_size,
        openclip_cachedir=openclip_cachedir,
        pretrained=pretrained,
    )