from collections import Counter

import fire
import numpy as np
import scipy.special
from tokenizers import decoders

from desk_util.path_helper import get_model_save_path, load_csv_dataset, load_clf_pred
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.bert_probe.probe_inference import ProbeInference



def extract_top_k_important(inf: ProbeInference, t):
    decoder = decoders.WordPiece()
    output = []
    item = inf.predict(t)
    inputs = inf.tokenizer(t)
    tokens = inf.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    cls_probs = scipy.special.softmax(item.bert_logits, axis=-1)[0]
    if cls_probs[1] < 0.5:
        return None

    probe_list = []
    indices = None
    for layer_no in range(len(item.layer_logits)):
        layer_logit = item.layer_logits[f"layer_{layer_no}"]
        layer_proba = scipy.special.softmax(layer_logit, axis=-1)

        scores = layer_proba[0, :len(tokens), 1]
        print("scores", scores.shape)
        bin = scores > 0.9
        if np.any(bin):
            bin = scores > 0.7

            indices = np.nonzero(bin)[0]
            print([scores[i] for i in indices])
            break

    important_list = []
    for idx, token in enumerate(tokens):
        if idx in indices:
            important_list.append(token)
        else:
            important_list.append(".")

    # print("Prediction: {}".format(round(cls_probs[1], 2)))
    important_s = decoder.decode(important_list)
    res = {
        "text": t,
        "important": important_s
    }
    return res


# Example usage:
def main(sb="askscience", model_name=""):
    if not model_name:
        model_name = f"bert2_{sb}"
        model_name = model_name + "_probe"
    model_path = get_model_save_path(model_name)
    dataset = f"{sb}_2_val_100"

    gpt_none_pred = load_clf_pred(dataset, "chatgpt_none")
    data = load_csv_dataset(dataset)
    labels = load_labels(dataset)

    counter = Counter()
    texts = []
    for idx in range(len(data)):
        data_id, label = labels[idx]
        assert gpt_none_pred[idx][0] == data_id
        gpt_pred = gpt_none_pred[idx][1]
        if gpt_pred == label:
            pass
        else:
            counter[(gpt_pred, label)] += 1
            texts.append(data[idx][1])

    print(counter)
    print("{} items wrong out of {}".format(len(texts), len(data)))
    texts = [t for t in texts if "thank you for submitting to" not in t]
    print("After excluding bot ones, {}".format(len(texts)))
    # texts = texts[:1]

    print("Loading model")
    inf = ProbeInference(model_path)
    for text in texts:
        item = extract_top_k_important(inf, text)
        if item is None:
            continue
        for k, v in item.items():
            print(f"<{k}>")
            print(v)
            print(f"</{k}>")



if __name__ == "__main__":
    fire.Fire(main)
