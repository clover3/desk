from typing import Any, Dict, List, Optional

from llamafactory.hparams import get_train_args
from llamafactory.train.callbacks import LogCallback
from llama_user.ft.lf_workflow import run_sft


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("Not supported: {}.".format(finetuning_args.stage))


def main():
    run_exp()


if __name__ == "__main__":
    main()

