from collections import Counter, defaultdict

import fire

from desk_util.path_helper import load_csv_dataset
from rule_gen.reddit.keyword_building.run3.ask_to_llama import load_rule_processed_json, load_feature_pred


def main(sb= "TwoXChromosomes"):
    dataset = f"{sb}_2_train_100"
    run_name = f"llama_rp_cq_{sb}"
    rule_name = "cluster_questions"
    text_d = dict(load_csv_dataset(dataset))
    q_list: list[str] = load_rule_processed_json(rule_name, sb)
    preds = load_feature_pred(run_name, dataset)

    per_f = defaultdict(dict)
    for d in preds:
        data_id = d["data_id"]
        for f_idx in range(len(d["result"])):
            per_f[f_idx][data_id] = d["result"][f_idx]

    f_idx = 2
    print(f_idx, q_list[f_idx])
    for text_id, text in text_d.items():
        if per_f[f_idx][text_id]:
            print("Text {text_id}: {text}".format(text_id=text_id, text=text))


if __name__ == "__main__":
    fire.Fire(main)