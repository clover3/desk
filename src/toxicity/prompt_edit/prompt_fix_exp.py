from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import read_csv
from chair.list_lib import left
from toxicity.llama_guard.output_convertor import parse_predictions
from toxicity.llama_helper.lf_client import LLMClient
from chair.misc_lib import SuccessCounter
from toxicity.path_helper import get_dataset_pred_save_path


# Iterate Prompt of Toxigen
# If wrong, put that into system prompt

def load_predictions_parsed(run_name, n_item):
    dataset_name = f"toxigen_train_head_{n_item}"
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    preds = read_csv(save_path)
    text_predictions = [e[1] for e in preds]
    raw_scores = [float(e[2]) for e in preds]
    # Convert to binary predictions
    target_string = "toxic"
    parsed: list[tuple[int, float]] = parse_predictions(text_predictions, raw_scores, target_string)
    return left(parsed)


def build_examples(test_dataset, preds):
    inst_examples = []
    n_item = len(preds)
    for i in range(n_item):
        label = test_dataset[i]["label"]
        text = test_dataset[i]["text"]
        if not label == preds[i]:
            label_str = ["benign", "toxic"][label]
            example_formatted = f"\n <example label={label_str}>{text}</example>"
            inst_examples.append(example_formatted)

    return "\n".join(inst_examples)


def main():
    n_item = 100
    preds = load_predictions_parsed("api_1", n_item)
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    test_dataset = list(test_dataset)[:n_item]

    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += " If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    instruction += "\n Here are examples \n"
    instruction += build_examples(test_dataset, preds)
    key_string = "toxic"
    client = LLMClient()
    edit_success = SuccessCounter()
    acc = SuccessCounter()
    for i in range(n_item):
        e = test_dataset[i]
        text = e["text"]
        ret = client.ask(text, instruction)
        label = int(key_string in ret)
        is_correct = e["label"] == label
        acc.add(is_correct)

        is_orig_correct = label == preds[i]
        if not is_orig_correct:
            is_correct = e["label"] == label
            edit_success.add(is_correct)

    print("Acc", acc.get_summary())
    print("edit_success", edit_success.get_summary())


if __name__ == "__main__":
    main()
