import torch
import logging

import fire
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator, Optional, Any

from desk_util.path_helper import get_model_log_save_dir_path, get_model_save_path
from rule_gen.reddit.path_helper import get_reddit_train_data_path, get_reddit_train_data_path_ex
from rule_gen.reddit.reddit_train_bert import prepare_datasets, build_training_argument, DataArguments

LOG = logging.getLogger("RedditTrainBert")

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from typing import List, Tuple
import numpy as np

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


class BertProbe(nn.Module):
    def __init__(
            self,
            bert_model: BertForSequenceClassification,
            hidden_size: int = 768,
            num_classes: int = 2,
            layers_to_probe: List[int] = None,
    ):
        super().__init__()

        self.bert = bert_model
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = hidden_size

        # If no specific layers are specified, probe all layers
        if layers_to_probe is None:
            self.layers_to_probe = list(range(self.num_layers + 1))  # +1 for embeddings
        else:
            self.layers_to_probe = layers_to_probe

        # Freeze BERT parameters
        self.bert.eval()  # Set BERT to evaluation mode
        for param in self.bert.parameters():
            param.requires_grad = False

        # Create separate probe layers for each BERT layer
        self.probes = nn.ModuleDict({
            f'layer_{layer}': nn.Linear(hidden_size, num_classes)
            for layer in self.layers_to_probe
        })

        # Final layer to combine predictions from all layers
        self.combination_layer = nn.Linear(
            len(self.layers_to_probe) * num_classes,
            num_classes
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # Run BERT with no_grad to prevent gradient flow
        with torch.no_grad():
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=True
            )

            # Detach hidden states to ensure no gradient flow back to BERT
            hidden_states = tuple(h.detach() for h in bert_outputs.hidden_states)
            bert_logits = bert_outputs.logits.detach()

        layer_logits = {}
        all_layer_predictions = []

        # Process each layer
        for layer_idx in self.layers_to_probe:
            layer_hidden = hidden_states[layer_idx] # [B, Seq, Hidden]
            # Get predictions for this layer
            layer_probe = self.probes[f'layer_{layer_idx}']
            layer_output = layer_probe(layer_hidden) # [B, Seq, num_classes]

            layer_logits[f'layer_{layer_idx}'] = layer_output
            all_layer_predictions.append(layer_output)

        # Combine predictions from all layers
        combined_predictions = torch.cat(all_layer_predictions, dim=-1)
        all_layer_logits = self.combination_layer(combined_predictions)
        return all_layer_logits, bert_logits, layer_logits

    def train(self, mode: bool = True):
        """
        Override train method to ensure BERT stays in eval mode
        """
        super().train(mode)
        self.bert.eval()
        return self


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
    bert_model = BertForSequenceClassification.from_pretrained(src_model_path, num_labels=num_labels)
    probe_model = BertProbe(
        bert_model=bert_model,
        hidden_size=768,  # BERT base hidden size
        num_classes=num_labels,
        layers_to_probe=layers_to_probe
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
    LOG.info(f"Saving trained probe to {final_model_dir}")
    torch.save(probe_model.state_dict(), f"{final_model_dir}/probe_model.pt")

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


# Example usage:
def main(sb="askscience", src_model_name=""):
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
    fire.Fire(main)