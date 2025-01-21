import logging
import numpy as np

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from transformers import Trainer

from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.base_bert.train_clf_common import get_compute_metrics
from rule_gen.reddit.train_common import compute_per_device_batch_size
from rule_gen.reddit.transfer.edit_exp import ClfEditorSpec
from toxicity.ee.edit_exp_runner import seed_everything

LOG = logging.getLogger("BertTransfer")


def pretty_dict(results, keys_to_keep):
    formatted = {k: round(results[k], 2) if isinstance(results[k], float) else results[k]
                 for k in keys_to_keep if k in results}

    return formatted

def build_training_argument(logging_dir, output_dir):
    # Set up training parameters
    num_train_epochs = 3
    learning_rate = 5e-5
    train_batch_size = 16
    eval_batch_size = 64
    warmup_ratio = 0.1
    # Load datasets to calculate total steps
    # Calculate total number of training steps
    per_device_batch_size = compute_per_device_batch_size(train_batch_size)
    LOG.info(f"Train/Per-device batch size: {train_batch_size}/{per_device_batch_size}")
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_dir=logging_dir,
    )
    return training_args



class BertTransfer(ClfEditorSpec):
    # Simple transfer learning strategy
    editor_name = "BT1"
    def __init__(self):
        self.model_name = "bert_train_mix3"
        model_path = get_model_save_path(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model_path = model_path
        self.max_length = 256


    def _get_tokenized_dataset(self, payload):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        dataset = Dataset.from_dict({
            'text': [x[0] for x in payload],
            'label': [x[1] for x in payload]
        })

        def tokenize_function(examples):
            return tokenizer(
                examples['text'], padding='max_length', truncation=True,
                max_length=self.max_length)

        tokenized = dataset.map(tokenize_function, batched=True)
        return tokenized

    def _augment_data(self, edit_payload):
        return edit_payload

    def batch_edit(self, edit_payload, edit_name=""):
        if not edit_name:
            edit_name = ""
        edit_name = edit_name + "_" + self.editor_name
        updated_model_name = "{}_{}".format(self.model_name, edit_name)
        logging_dir = get_model_log_save_dir_path(updated_model_name)
        output_dir = get_model_save_path(updated_model_name)
        training_args = build_training_argument(logging_dir, output_dir)
        training_args.num_train_epochs = 4
        train_payload = self._augment_data(edit_payload)  # Train payload might be more than edit_payload
        tokenized_train = self._get_tokenized_dataset(train_payload)
        seed_everything(42)

        # Run training
        LOG.info("Initializing Trainer")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_train,
            compute_metrics=get_compute_metrics(),
        )
        # Train the model
        LOG.info("Starting model training")
        train_result = trainer.train()
        # train_result = pretty_dict(train_result,  ['train_runtime', 'train_loss'])
        LOG.info('train_result %s', str(train_result))
        LOG.info("Model training completed")
        tokenized_edit = self._get_tokenized_dataset(edit_payload)
        eval_results = trainer.evaluate(tokenized_edit)
        LOG.info("Eval on training split", eval_results)

    def predict(self, eval_payload: list[tuple[str, int]] | list[str]):
        if not eval_payload:
            return []
        if type(eval_payload[0]) is str:
            eval_payload = [(t, 0) for t in eval_payload]

        args = TrainingArguments(output_dir="tmp_trainer", disable_tqdm=True)
        trainer = Trainer(model=self.model, args=args)
        tokenized_eval = self._get_tokenized_dataset(eval_payload)
        raw_predictions = trainer.predict(tokenized_eval)
        predictions = raw_predictions.predictions
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels


class BertTransfer2(BertTransfer):
    editor_name = "BT2"  # Simply define the name as class variable

    def __init__(self, aux_data):
        self.aux_data = aux_data
        super(BertTransfer2, self).__init__()

    def _augment_data(self, edit_payload):
        return edit_payload + self.aux_data

