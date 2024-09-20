
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.train.callbacks import PissaConvertCallback, SaveProcessorCallback
from llamafactory.train.trainer_utils import create_custom_optimizer, create_custom_scheduler
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


def _create_desk_badam_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    named_parameters = model.named_parameters()
    trainable_layers: list[str] = get_trainable_layers(model, finetuning_args)
    trainable_parameters = filter_trainable_param(named_parameters, trainable_layers)

    decay_params, nodecay_params = [], []
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in trainable_parameters:
        if param.requires_grad:
            if name in decay_param_names:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=nodecay_params, weight_decay=0.0),
        dict(params=decay_params, weight_decay=training_args.weight_decay),
    ]

    if finetuning_args.badam_mode == "layer":
        from badam import BlockOptimizer

        base_optimizer = optim_class(param_groups, **optim_kwargs)
        optimizer = BlockOptimizer(
            base_optimizer=base_optimizer,
            named_parameters_list=list(named_parameters),
            block_prefix_list=None,
            switch_block_every=finetuning_args.badam_switch_interval,
            start_block=finetuning_args.badam_start_block,
            switch_mode=finetuning_args.badam_switch_mode,
            verbose=finetuning_args.badam_verbose,
            ds_zero3_enabled=is_deepspeed_zero3_enabled(),
        )
        logger.info(
            f"Using BAdam optimizer with layer-wise update, switch mode is {finetuning_args.badam_switch_mode}, "
            f"switch block every {finetuning_args.badam_switch_interval} steps, "
            f"default start block is {finetuning_args.badam_start_block}"
        )

    elif finetuning_args.badam_mode == "ratio":
        from badam import BlockOptimizerRatio

        assert finetuning_args.badam_update_ratio > 1e-6
        optimizer = BlockOptimizerRatio(
            param_groups=param_groups,
            named_parameters_list=list(model.named_parameters()),
            update_ratio=finetuning_args.badam_update_ratio,
            mask_mode=finetuning_args.badam_mask_mode,
            verbose=finetuning_args.badam_verbose,
            include_embedding=False,
            **optim_kwargs,
        )
        logger.info(
            f"Using BAdam optimizer with ratio-based update, update ratio is {finetuning_args.badam_update_ratio}, "
            f"mask mode is {finetuning_args.badam_mask_mode}"
        )

    return optimizer


def filter_trainable_param(named_parameters, trainable_layers):
    trainable_parameters = []
    for name, param in named_parameters:
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
                forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
            trainable_parameters.append(param)
    return trainable_parameters


def create_desk_custom_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:
    if finetuning_args.use_badam:
        return _create_desk_badam_optimizer(model, training_args, finetuning_args)
    else:
        return create_custom_optimizer(model, training_args, finetuning_args)


class DeskTrainer(CustomSeq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_desk_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()
