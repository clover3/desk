import logging
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Optional
import evaluate
import fire
from transformers import Trainer, TrainingArguments, BertTokenizer

from chair.misc_lib import rel
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.base_bert.reddit_train_bert import build_training_argument, DataArguments
from rule_gen.reddit.base_bert.reddit_train_bert import prepare_datasets
from rule_gen.reddit.bert_c.c2_modeling import C2Config, BertC2
from rule_gen.reddit.bert_c.load_macro_norm_violation import load_norm_id_mapping
from rule_gen.reddit.bert_c.macro_norm_aug import load_triplet_and_norm_dataset, prepare_norm_dataset
from rule_gen.reddit.bert_c.train_w_sb_ids import load_sb_name_to_id_mapping
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper3 import JobContext

LOG = logging.getLogger("TrainC2")


def get_compute_metrics():
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        b_pred = logits > 0
        predictions = b_pred.astype(int)
        return clf_metrics.compute(predictions=predictions, references=labels)
    return compute_metrics


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    train_data_path: str = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    max_length: int = field(default=256)
    debug: bool = field(default=False)
    n_policy: int = field(default=72)



def prepare_datasets(dataset_args: DataArguments, model_name):
    sb_name_dict = load_sb_name_to_id_mapping()
    norm_dict = load_norm_id_mapping()
    LOG.info(f"Loading training data from {rel(dataset_args.train_data_path)}")
    norm_data = prepare_norm_dataset("train", norm_dict, dataset_args.n_policy)
    train_dataset = load_triplet_and_norm_dataset(dataset_args.train_data_path, norm_data, sb_name_dict)
    train_dataset.shuffle()

    LOG.info(f"Loading evaluation data from {rel(dataset_args.eval_data_path)}")
    norm_data = prepare_norm_dataset("val", norm_dict, dataset_args.n_policy)
    eval_dataset = load_triplet_and_norm_dataset(dataset_args.eval_data_path, norm_data, sb_name_dict)
    eval_dataset.shuffle()


    LOG.info("Creating datasets")
    if dataset_args.debug:
        LOG.info("Debug mode on. Take 100 items")
        train_dataset = train_dataset.take(100)
        eval_dataset = eval_dataset.take(100)

    # Load tokenizer and model
    LOG.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], padding='max_length',
            truncation=True, max_length=dataset_args.max_length)

    num_proc = multiprocessing.cpu_count()
    LOG.info(f"Using {num_proc} processes for dataset mapping")
    LOG.info("Tokenizing training dataset")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    LOG.info('tokenized_train.features {}'.format(tokenized_train.features))
    LOG.info("Tokenizing evaluation dataset")
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    return tokenized_train, tokenized_eval


def train_c(
        model_name,
        model_factory,
        training_args: TrainingArguments,
        dataset_args,
        final_model_dir: str,
) -> Dict[str, float]:
    LOG.info("Starting training process")
    tokenized_train, tokenized_eval = prepare_datasets(dataset_args, model_name)
    model = model_factory()
    training_args.include_inputs_for_metrics = True
    training_args.label_names = ["labels"]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=get_compute_metrics(),
    )

    LOG.info("Starting training")
    train_result = trainer.train()
    LOG.info("Training completed")

    eval_results = trainer.evaluate(tokenized_eval)
    LOG.info("Evaluation results:{}".format(eval_results))

    trainer.save_model(training_args.output_dir)

    LOG.info(f"Training outputs and logs are saved in: {training_args.output_dir}")
    LOG.info(f"Final fine-tuned model and tokenizer are saved in: {final_model_dir}")
    LOG.info("BERT fine-tuning process completed")
    return eval_results


def run_train(base_model, save_model_name, model_factory, n_policy, debug):
    init_logging()
    sb = "train_comb4"
    data_name = "train_data2"
    with JobContext(save_model_name + "_train"):
        output_dir = get_model_save_path(save_model_name)
        final_model_dir = get_model_save_path(save_model_name)
        logging_dir = get_model_log_save_dir_path(save_model_name)
        max_length = 256
        training_args = build_training_argument(logging_dir, output_dir, debug)
        training_args.num_train_epochs = 1
        dataset_args = DataArguments(
            train_data_path=get_reddit_train_data_path_ex(
                data_name, sb, "train"),
            eval_data_path=get_reddit_train_data_path_ex(
                data_name, sb, "val"),
            max_length=max_length,
            debug=debug,
            n_policy=n_policy,
        )

        eval_result = train_c(
            model_name=base_model,
            model_factory=model_factory,
            training_args=training_args,
            dataset_args=dataset_args,
            final_model_dir=final_model_dir,
        )
        #
        # proxy = get_task_manager_proxy()
        # for metric in ["eval_f1"]:
        #     dataset = sb + "_val"
        #     metric_short = metric[len("eval_"):]
        #     proxy.report_number(save_model_name, eval_result[metric], dataset, metric_short)


def main(debug=False):
    model_name = f"bert_c2_2"
    c_config = C2Config()

    def model_factory():
        model = BertC2(c_config)
        return model

    run_train(c_config.base_model_name,
              model_name, model_factory, c_config.n_policy, debug)


if __name__ == "__main__":
    fire.Fire(main)
