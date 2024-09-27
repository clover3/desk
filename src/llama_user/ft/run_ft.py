from typing import Any, Dict, List, Optional

from llamafactory.hparams import get_train_args
from llamafactory.train.callbacks import LogCallback
from llamafactory.train.sft import run_sft
from toxicity.path_helper import get_model_save_path


def run_exp(overwrite_args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("Not supported: {}.".format(finetuning_args.stage))



def main():
    run_name = "ft14"
    args = {
        "output_dir": get_model_save_path(run_name),
        "dataset": "toxigen"
    }
    run_exp(args)


if __name__ == "__main__":
    main()

