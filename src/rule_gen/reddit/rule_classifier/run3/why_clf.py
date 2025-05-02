from desk_util.clf_util import eval_prec_recall_f1_acc
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
import os
import json
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import fire
from sklearn.metrics import confusion_matrix


def get_value(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    fp, tp = m[:, 1]
    return tp / (tp + fp), tp


def main(sb):
    dataset_fmt = "{}_2_val_100"
    run_name_fmt = "chatgpt_why_{}".format(sb)
    dataset = dataset_fmt.format(sb)
    run_name = run_name_fmt.format(sb)

    res_path = os.path.join(
        output_root_path, "reddit",
        "j_res", dataset, f"{run_name}.json")
    labels = load_labels(dataset)

    j = json.load(open(res_path, "r"))
    arr = []
    reps = []
    y = []
    for e, label in zip(j, labels):
        data_id, pred, score, text = e
        tokens = word_tokenize(text.lower())
        counter = Counter(tokens)
        reps.append(counter)

        x_i = tokens[0] == "yes" and "hate" not in tokens
        # x_i = tokens[0] == "yes"
        arr.append(x_i)
        data_id_, label_s = label
        y.append(int(label_s))

    v = get_value(y, arr)
    ret = eval_prec_recall_f1_acc(y, arr)
    print("Train", v)
    print(ret)



if __name__ == "__main__":
    fire.Fire(main)
