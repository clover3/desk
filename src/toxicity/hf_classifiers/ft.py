import os
import math
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from toxicity.dataset_helper.csv_datasets import load_csv_as_hf_dataset
from toxicity.path_helper import get_model_save_path


def main():
    run_name = "s-nlp_1"
    dataset = "toxigen_train_head_100"
    model_name = "s-nlp/roberta_toxicity_classifier"

    run_ft(dataset, model_name, run_name)


def run_ft(dataset, model_name, run_name):
    output_dir = get_model_save_path(run_name)
    log_path = os.path.join(output_dir, "logs")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 128  # You can adjust this value based on your needs

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length",
                         max_length=max_length,
                         truncation=True)

    #note
    train_dataset = load_csv_as_hf_dataset(dataset)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    # Calculate total number of training steps
    num_epochs = 10
    batch_size = 8
    dataset_size = len(train_dataset)
    print("dataset_size", dataset_size)
    total_steps = math.ceil(dataset_size / batch_size) * num_epochs
    # Set warmup_steps to 20% of total steps
    warmup_steps = max(int(0.2 * total_steps), 3)
    print("warmup_steps", warmup_steps)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=log_path,
        logging_steps=100,
        save_strategy="no",  # Disable saving during training
        report_to="none",  # Disable reporting to avoid creating unnecessary files
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()