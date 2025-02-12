import logging

import fire
from rule_gen.reddit.keyword_building.run3.ask_to_llama import apply_fn_to_dataset, get_apply_fn

LOG = logging.getLogger(__name__)


def apply_fn_main(
        run_name: str,
        dataset: str,
        overwrite=False,
) -> None:
    predict_fn = get_apply_fn(run_name)
    apply_fn_to_dataset(dataset, run_name, predict_fn, overwrite)


if __name__ == "__main__":
    fire.Fire(apply_fn_main)
