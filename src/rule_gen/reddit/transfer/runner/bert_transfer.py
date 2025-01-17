import logging
import numpy as np

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer

from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.base_bert.train_clf_common import get_compute_metrics
from rule_gen.reddit.reddit_train_bert import build_training_argument
from rule_gen.reddit.transfer.edit_exp import ClfEditorSpec
from toxicity.ee.edit_exp_runner import seed_everything

LOG = logging.getLogger(__name__)


class BertTransfer(ClfEditorSpec):
    def __init__(self, edit_name):
        self.edit_name = edit_name
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

    def batch_edit(self, edit_payload, edit_name=""):
        if not edit_name:
            edit_name = ""
        updated_model_name = "{}_{}".format(self.model_name, edit_name)
        logging_dir = get_model_log_save_dir_path(updated_model_name)
        output_dir = get_model_save_path(updated_model_name)
        training_args = build_training_argument(logging_dir, output_dir)
        training_args.num_train_epochs = 4
        train_payload = edit_payload  # Train payload might be more than edit_payload
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
        print('train_result', train_result)
        LOG.info("Model training completed")
        tokenized_edit = self._get_tokenized_dataset(edit_payload)
        eval_results = trainer.evaluate(tokenized_edit)
        print("Edit payload", eval_results)

    def predict(self, eval_payload):
        trainer = Trainer(model=self.model)
        tokenized_eval = self._get_tokenized_dataset(eval_payload)
        raw_predictions = trainer.predict(tokenized_eval)
        predictions = raw_predictions.predictions
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels
