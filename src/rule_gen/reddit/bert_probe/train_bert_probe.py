import fire

from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_log_save_dir_path, get_model_save_path
from rule_gen.reddit.bert_probe.train import train_probe
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.reddit.base_bert.reddit_train_bert import build_training_argument, DataArguments


# Example usage:
def do_train_bert_probe(sb="askscience", src_model_name=""):
    max_length = 256

    if not src_model_name:
        src_model_name = f"bert2_{sb}"
    new_model_name = src_model_name + "_probe"
    src_model_path = get_model_save_path(src_model_name)

    output_dir = get_model_save_path(new_model_name)
    logging_dir = get_model_log_save_dir_path(new_model_name)
    training_args = build_training_argument(logging_dir, output_dir)
    training_args.metric_for_best_model = "probe_accuracy"
    training_args.num_train_epochs = 1
    training_args.eval_steps = 100

    dataset_args = DataArguments(
        train_data_path=get_reddit_train_data_path_ex("train_data2", sb, "train"),
        eval_data_path=get_reddit_train_data_path_ex("train_data2", sb, "val"),
        max_length=max_length
    )
    eval_results = train_probe(
        src_model_path=src_model_path,
        training_args=training_args,
        dataset_args=dataset_args,
        final_model_dir=output_dir,
        num_labels=2,
    )


if __name__ == "__main__":
    init_logging()
    fire.Fire(do_train_bert_probe)