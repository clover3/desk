import logging
from typing import Dict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainingArguments, AutoConfig

from rule_gen.reddit.bert_probe.probe_model import BertProbe
from rule_gen.reddit.base_bert.reddit_train_bert import prepare_datasets

LOG = logging.getLogger("RedditTrainBert")

def get_compute_metrics():
    def compute_metrics(eval_pred):
        final_logits = eval_pred.predictions[0]  # Final probe predictions
        bert_logits = eval_pred.predictions[1]  # BERT's predictions
        layer_logits_dict = eval_pred.predictions[2]  # Layer-specific predictions

        # Convert logits to predictions
        final_preds = np.argmax(final_logits, axis=-1)
        bert_preds = np.argmax(bert_logits, axis=1)
        bert_preds_ex = np.expand_dims(bert_preds, 1)
        # Calculate final accuracy (how well probe matches BERT)
        final_accuracy = (final_preds == bert_preds_ex).mean()
        # Calculate layer-specific accuracies
        layer_accuracies = {
            name: (np.argmax(logits, axis=-1) == bert_preds_ex).mean()
            for name, logits in layer_logits_dict.items()
        }

        # Combine all metrics
        metrics = {
            "probe_accuracy": final_accuracy,
            **{f"{name}_accuracy": acc for name, acc in layer_accuracies.items()}
        }

        return metrics

    return compute_metrics


class ProbeTrainer(Trainer):
    """Custom Trainer for the probe model to use BERT's predictions as targets"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get attention mask if it exists in inputs
        attention_mask = inputs.get("attention_mask", None)

        # Forward pass through the probe model
        final_logits, bert_logits, layer_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            token_type_ids=inputs.get("token_type_ids", None)
        )

        with torch.no_grad():
            targets = torch.argmax(bert_logits, dim=-1)

        # Initialize loss function (we'll apply the mask manually)
        loss_fct = CrossEntropyLoss(reduction='none')

        def apply_loss(logits):
            # Calculate final loss with mask
            B, L, C = logits.shape
            targets_ex = torch.tile(targets.unsqueeze(1), [1, L])
            logits_flat = torch.reshape(logits, [-1, C])
            loss_flat = loss_fct(logits_flat, targets_ex.flatten())
            loss_per_seq = torch.reshape(loss_flat, [B, L])
            loss_per_sample = (loss_per_seq * attention_mask.float()).sum(dim=1)
            return loss_per_sample
            # Average only over non-padded tokens in each batch

        final_loss = apply_loss(final_logits)

        # Calculate layer-specific losses with mask
        layer_losses = {}
        for name, logits in layer_logits.items():
            layer_loss = apply_loss(logits)
            layer_losses[name] = layer_loss

        total_loss_per_sample = final_loss + sum(layer_losses.values())
        total_loss = total_loss_per_sample.mean()
        if return_outputs:
            outputs = {
                "final_logits": final_logits,
                "bert_logits": bert_logits,
                "layer_logits": layer_logits,
                "layer_losses": layer_losses,
                "final_loss": final_loss
            }
            return total_loss, outputs

        return total_loss


def train_probe(
        src_model_path: str,
        training_args: TrainingArguments,
        dataset_args,
        final_model_dir: str,
        num_labels: int = 2,
        layers_to_probe: list = None
) -> Dict[str, float]:
    LOG.info("Starting probe training process")
    tokenized_train, tokenized_eval = prepare_datasets(dataset_args, src_model_path)
    probe_model = BertProbe.from_pretrained(
        src_model_path,
        hidden_size=768,  # BERT base hidden size
        num_classes=num_labels,
        layers_to_probe=layers_to_probe,
        config=AutoConfig.from_pretrained(src_model_path)
    )

    # Initialize ProbeTrainer
    LOG.info("Initializing ProbeTrainer")
    trainer = ProbeTrainer(
        model=probe_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=get_compute_metrics(),
    )

    # Train the probe
    LOG.info("Starting probe training")
    train_result = trainer.train()
    LOG.info("Probe training completed")

    # Evaluate
    eval_results = trainer.evaluate(tokenized_eval)
    print("Evaluation results:", eval_results)

    # Save the probe model
    trainer.save_model(training_args.output_dir)

    # Save probe configuration
    probe_config = {
        "hidden_size": 768,
        "num_classes": num_labels,
        "layers_to_probe": layers_to_probe,
        "src_model_name": src_model_path
    }
    import json
    with open(f"{final_model_dir}/probe_config.json", "w") as f:
        json.dump(probe_config, f)

    LOG.info(f"Training outputs and logs are saved in: {training_args.output_dir}")
    LOG.info(f"Final probe model is saved in: {final_model_dir}")
    LOG.info("Probe training process completed")

    return eval_results
