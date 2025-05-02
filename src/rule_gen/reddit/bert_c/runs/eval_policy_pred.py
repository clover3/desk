import numpy as np
import pandas as pd
import torch
import torch
from datasets import Dataset

from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_c.c2_modeling import BertC2, C2Output
from rule_gen.reddit.bert_c.load_macro_norm_violation import load_norm_id_mapping
from rule_gen.reddit.bert_c.macro_norm_aug import prepare_norm_dataset
from rule_gen.reddit.bert_c.train_w_sb_ids import load_sb_name_to_id_mapping

def get_bert_c_predictor(model_cls, run_name):
    model_name, sb_name = run_name.split("/")
    max_length = 256
    model_path = get_model_save_path(model_name)
    model = model_cls.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to appropriate device
    sb_name_dict = load_sb_name_to_id_mapping()
    sb_id = sb_name_dict[sb_name]

    def predict(text):
        inputs = model.tokenizer(
            text, padding='max_length',
            truncation=True, max_length=max_length, return_tensors="pt")
        inputs["sb_id"] = torch.tensor([sb_id], dtype=torch.long)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            c2_output: C2Output = model(**inputs, return_dict=True)
        return c2_output

    return predict


def get_dataset():
    n_policy = 72
    sb_name_dict = load_sb_name_to_id_mapping()
    norm_dict = load_norm_id_mapping()
    norm_data = prepare_norm_dataset("val", norm_dict, n_policy)
    norm_df = pd.DataFrame(norm_data)

    n_policy = len(norm_df["policy_labels"][0])
    zeros = [0] * n_policy
    norm_df['sb_name'] = ['askscience'] * len(norm_data)
    norm_df['sb_id'] = norm_df['sb_name'].map(sb_name_dict)
    dataset = Dataset.from_pandas(norm_df)
    dataset = dataset.shuffle()
    return dataset


def apply_model():
    model_cls = BertC2  # Your model class
    run_name = "bert_c2_2/_unknown_"
    model_name, sb_name = run_name.split("/")
    dataset = get_dataset()
    dataset = dataset.take(30)
    model_path = get_model_save_path(model_name)
    model = model_cls.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to appropriate device
    sb_name_dict = load_sb_name_to_id_mapping()
    sb_id = sb_name_dict[sb_name]
    max_length = 256

    for batch in dataset:
        tokenized = model.tokenizer(
            batch['text'], padding='max_length',
            truncation=True, max_length=max_length)

        for k, v in batch.items():
            if type(v) != str:
                tokenized[k] = v

        tokenized["sb_id"] = sb_id
        tokenized = {k: [v] for k, v in tokenized.items()}
        inputs = {k: torch.tensor(v).to(device) for k, v in tokenized.items()}
        with torch.no_grad():
            c2_output = model(**inputs, return_dict=True)

        print(c2_output)



def evaluate_model(model_cls, run_name, dataset):
    predictor = get_bert_c_predictor(model_cls, run_name)
    d = load_norm_id_mapping()
    rev_d = {v:k for k,v in d.items()}
    rev_d[0] = "notused"

    # Store predictions and true labels
    all_predictions = []
    all_scores = []
    all_true_labels = dataset["policy_labels"]

    # Make predictions for each text
    for text, policy_label in zip(dataset["text"], dataset["policy_labels"]):
        c2_output: C2Output = predictor(text)
        print(text)
        binary_pred = c2_output.policy_pred > 0
        binary_pred_np = binary_pred.int().cpu().numpy()
        pw = c2_output.policy_pred * c2_output.w
        pw = pw.cpu().numpy()
        gen = pw[:, :8]
        spe = pw[:, 8:]
        print("gen, spe: ", np.mean(gen), np.mean(spe))
        all_predictions.append(binary_pred_np)
        out_s = []
        for i, v in enumerate(binary_pred_np[0,:8]):
            if v:
                out_s.append(rev_d[i])
        gold_i = np.argmax(policy_label[:8])
        print(out_s)
        print(binary_pred_np[0,:8])
        print("label", rev_d[gold_i])

        all_scores.append(c2_output.policy_pred.cpu().numpy())

    # Convert to tensors for easier computation
    predictions = np.stack(all_predictions, axis=0)
    scores = np.stack(all_scores, axis=0)
    true_labels = np.array(all_true_labels)

    print("predictions", predictions.shape)
    print("true_labels", true_labels.shape)
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # Convert tensors to numpy arrays
    predictions = predictions[:, :, :8]
    true_labels = true_labels[:, :8]

    print("predictions", predictions)
    print("true_labels_", true_labels)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels.flatten(), predictions.flatten(), average='weighted'
    )
    # Return metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'scores': scores.tolist()
    }

    return metrics


def main():
    model_cls = BertC2  # Your model class
    run_name = "bert_c2_1/_unknown_"  # Replace with actual model/subreddit
    dataset = get_dataset()
    metrics = evaluate_model(model_cls, run_name, dataset.take(30))

    # Print metrics
    print(f"Evaluation Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    return metrics


if __name__ == "__main__":
    apply_model()