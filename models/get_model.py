from models.resnet_micro import (
    resnet18_cifar_stem,
    resnet18_imagenet_stem,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
    resnext101_32x8d,
    resnext101_64x4d,
)
from models.vitcifar import vit_small, vit_tiny, vit_base, vit_b_32_
from models.clip_vit import get_CLIPclassification_model
from models.mlp import create_mlp


def _create_mlp_model(pretrained=False, input_size=12288, num_classes=200, **kwargs):
    mlp_defaults = {
        "hidden_sizes": [4096, 2048, 1024],
        "dropout_rate": 0.5,
        "use_bn": True,
    }
    mlp_defaults.update(kwargs)
    return create_mlp(input_size=input_size, num_classes=num_classes, **mlp_defaults)


def _create_clip_model(
    pretrained=False,
    model_name="ViT-B-32",
    freeze_clip=False,
    **kwargs,
):
    num_classes = kwargs.pop("num_classes", 200)

    # OpenCLIP warns when OpenAI ViT-B-32 weights are loaded into a non-quickgelu config.
    # Keep user overrides intact; only apply the safer default upgrade path.
    if model_name == "ViT-B-32" and bool(pretrained):
        model_name = "ViT-B-32-quickgelu"

    return get_CLIPclassification_model(
        model_name=model_name,
        num_classes=num_classes,
        freeze_clip=freeze_clip,
        pretrained=pretrained,
        **kwargs,
    )


def _create_clip_model_cifar100(pretrained=False, **kwargs):
    kwargs.setdefault("num_classes", 100)
    return _create_clip_model(pretrained=pretrained, model_name="ViT-B-32", **kwargs)


def _create_clip_model_default(pretrained=False, **kwargs):
    kwargs.setdefault("num_classes", 200)
    return _create_clip_model(pretrained=pretrained, model_name="ViT-B-32", **kwargs)


def _create_clip_model_default_frozen(pretrained=False, **kwargs):
    kwargs.setdefault("num_classes", 200)
    return _create_clip_model(
        pretrained=pretrained,
        model_name="ViT-B-32",
        freeze_clip=True,
        **kwargs,
    )


MODEL_REGISTRY = {
    "resnet18": resnet18_cifar_stem,
    "resnet18_cifar_stem": resnet18_cifar_stem,
    "resnet18_imagenet_stem": resnet18_imagenet_stem,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "resnext50_32x4d": resnext50_32x4d,
        "resnext101_32x8d": resnext101_32x8d,
        "resnext101_64x4d": resnext101_64x4d,
        "wide_resnet50_2": wide_resnet50_2,
        "wide_resnet101_2": wide_resnet101_2,
        "vit_small": vit_small,
        "vit_tiny": vit_tiny,
        "vit_base": vit_base,
        "vit_b_32": vit_b_32_,
        "mlp": _create_mlp_model,
        "clip_pretrained_vit_b_32_cifar100": _create_clip_model_cifar100,
        # "clip_pretrained_vit_base_cifar100_freeze": lambda pretrained: get_CLIPclassification_model(
        #     model_name="ViT-B/32", nb_clases=100, freeze_clip=True
        # ),
        "clip_vit": _create_clip_model_default,
        "clip_vit_frozen": _create_clip_model_default_frozen,
    }


def get_model(model_name, pretrained=False, **model_kwargs):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_REGISTRY[model_name](pretrained=pretrained, **model_kwargs)
