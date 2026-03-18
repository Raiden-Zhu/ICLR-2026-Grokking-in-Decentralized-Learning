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


CLIP_MODEL_NAMES = {
    "clip_pretrained_vit_b_32_cifar100",
    "clip_vit",
    "clip_vit_frozen",
}


def _create_mlp_model(pretrained=False, input_size=12288, num_classes=200, **kwargs):
    mlp_defaults = {
        "hidden_sizes": [4096, 2048, 1024],
        "dropout_rate": 0.5,
        "use_bn": True,
    }
    mlp_defaults.update(kwargs)
    return create_mlp(input_size=input_size, num_classes=num_classes, **mlp_defaults)


def _resolve_clip_model_name(model_name, pretrained):
    # OpenCLIP warns when OpenAI ViT-B-32 weights are loaded into a non-quickgelu config.
    # Keep user overrides intact; only apply the safer default upgrade path.
    if model_name == "ViT-B-32" and bool(pretrained):
        return "ViT-B-32-quickgelu"
    return model_name


def _create_clip_model(
    pretrained=False,
    model_name="ViT-B-32",
    freeze_clip=False,
    schema_only=False,
    **kwargs,
):
    num_classes = kwargs.pop("num_classes", 200)
    resolved_model_name = _resolve_clip_model_name(model_name, pretrained)
    effective_pretrained = None if schema_only else pretrained

    return get_CLIPclassification_model(
        model_name=resolved_model_name,
        num_classes=num_classes,
        freeze_clip=freeze_clip,
        pretrained=effective_pretrained,
        **kwargs,
    )


def _create_clip_model_cifar100(pretrained=False, schema_only=False, **kwargs):
    kwargs.setdefault("num_classes", 100)
    return _create_clip_model(
        pretrained=pretrained,
        model_name="ViT-B-32",
        schema_only=schema_only,
        **kwargs,
    )


def _create_clip_model_default(pretrained=False, schema_only=False, **kwargs):
    kwargs.setdefault("num_classes", 200)
    return _create_clip_model(
        pretrained=pretrained,
        model_name="ViT-B-32",
        schema_only=schema_only,
        **kwargs,
    )


def _create_clip_model_default_frozen(pretrained=False, schema_only=False, **kwargs):
    kwargs.setdefault("num_classes", 200)
    return _create_clip_model(
        pretrained=pretrained,
        model_name="ViT-B-32",
        freeze_clip=True,
        schema_only=schema_only,
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


def get_model(model_name, pretrained=False, schema_only=False, **model_kwargs):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")

    factory = MODEL_REGISTRY[model_name]
    if schema_only and model_name in CLIP_MODEL_NAMES:
        return factory(pretrained=pretrained, schema_only=True, **model_kwargs)
    if schema_only:
        return factory(pretrained=False, **model_kwargs)
    return factory(pretrained=pretrained, **model_kwargs)
