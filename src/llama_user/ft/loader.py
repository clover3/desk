
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from llamafactory.model.adapter import init_adapter
from llamafactory.model.model_utils.misc import register_autoclass
from llamafactory.model.model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from llamafactory.model.model_utils.unsloth import load_unsloth_pretrained_model
from llamafactory.model.model_utils.valuehead import load_valuehead_params
from llamafactory.model.model_utils.visual import get_image_seqlen
from llamafactory.model.patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model


def get_trainable_layers(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
) -> list[str]:

    logger.info("Fine-tuning method: Freeze")
    if hasattr(model.config, "text_config"):  # composite models
        config = getattr(model.config, "text_config")
    else:
        config = model.config

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                "`num_layers` {} should be divisible by `num_layer_trainable` {}.".format(
                    num_layers, finetuning_args.freeze_trainable_layers
                )
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)
    return trainable_layers


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantized models can only be used for the LoRA tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 and not fsdp (zero3 or fsdp already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and (is_deepspeed_zero3_enabled() or is_fsdp_enabled()):
        logger.info("ZeRO3 / FSDP detected, remaining trainable params in float32.")
    else:
        logger.info("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type in ["full", "freeze"]:
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError("Unknown finetuning type: {}.".format(finetuning_args.finetuning_type))

    return model


def load_model_desk_ver(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config)
            else:
                model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model
