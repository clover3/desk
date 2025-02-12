import fire
import numpy as np
import scipy.special
from tokenizers import decoders

from chair.list_lib import right
from desk_util.path_helper import get_model_save_path, load_csv_dataset
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.bert_probe.probe_inference import ProbeInference


def extract_top_k_important(inf: ProbeInference, texts):
    decoder = decoders.WordPiece()
    output = []
    for t in texts:
        item = inf.predict(t)
        inputs = inf.tokenizer(t)
        tokens = inf.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        cls_probs = scipy.special.softmax(item.bert_logits, axis=-1)[0]
        if cls_probs[1] < 0.5:
            continue

        probe_list = []
        for layer_no in range(len(item.layer_logits)):
            layer_logit = item.layer_logits[f"layer_{layer_no}"]
            probe_list.append(layer_logit)

        probe_concat = np.stack(probe_list, axis=2)
        probe = np.mean(probe_concat, axis=2)
        probe = probe[0, :len(tokens), 1]
        rank = np.argsort(probe)[::-1]

        k = int(len(probe) * 0.3)
        if len(probe) < 10:
            k = int(len(probe) * 0.5)

        top_k = rank[:k]

        important_list = []
        for idx, token in enumerate(tokens):
            if idx in top_k:
                important_list.append(token)
            else:
                important_list.append(".")

        # print("Prediction: {}".format(round(cls_probs[1], 2)))
        important_s = decoder.decode(important_list)
        res = {
            "text": t,
            "important": important_s
        }
        output.append(res)

    return output


# Example usage:
def main(sb="askscience", model_name=""):
    if not model_name:
        model_name = f"bert2_{sb}"
        model_name = model_name + "_probe"
    model_path = get_model_save_path(model_name)
    dataset = f"{sb}_2_val_100"
    data = load_csv_dataset(dataset)
    labels = load_labels(dataset)
    texts = right(data)[:200]
    inf = ProbeInference(model_path)

    output = extract_top_k_important(inf, texts)
    for k, v in output:
        print(f"<{k}>")
        print(v)
        print(f"</{k}>")


if __name__ == "__main__":
    fire.Fire(main)
