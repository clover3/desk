import logging
import os

from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.path_helper import get_split_subreddit_list

LOG = logging.getLogger(__name__)


def main():
    init_logging()
    for sb in get_split_subreddit_list("train"):
        model_name = f"bert2_{sb}"
        output_dir = get_model_save_path(model_name)

        l = [os.path.join(output_dir),
             os.path.join(output_dir, "model.safetensors")]
        for p in l:
            if os.path.exists(p):
                print(f"{p} Exists")


if __name__ == "__main__":
    main()
