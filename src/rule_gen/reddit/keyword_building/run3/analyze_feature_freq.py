from collections import Counter

import fire

from rule_gen.reddit.keyword_building.run3.ask_to_llama import load_rule_processed_json, load_feature_pred


def main(sb= "TwoXChromosomes"):
    dataset = f"{sb}_2_train_100"
    run_name = f"llama_rp_cq_{sb}"
    rule_name = "cluster_questions"
    q_list: list[str] = load_rule_processed_json(rule_name, sb)
    preds = load_feature_pred(run_name, dataset)

    counter = Counter()
    for d in preds:
        for f_idx in range(len(d["result"])):
            f = d["result"][f_idx]
            counter[f_idx, f] += 1

    for f_idx in range(len(d["result"])):
        n_true = counter[f_idx, 1]
        n_false = counter[f_idx, 0]
        n_rate = n_true / (n_true + n_false)
        print(f_idx, "{0:.2f}".format(n_rate), q_list[f_idx])


if __name__ == "__main__":
    fire.Fire(main)