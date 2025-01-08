import json
import os
from typing import Callable

import fire

from rule_gen.cpath import output_root_path
from rule_gen.open_ai_mod.path_helper import get_rule_gen_save_path
from rule_gen.text_gen.text_sim import TextSimilarity
from taskman_client.task_proxy import get_task_manager_proxy


def load_open_ai_category_labels() -> list[str]:
    p = os.path.join(output_root_path, "open_ai_mod", "categories.json")
    categories = json.load(open(p, "r"))["categories"]
    return [t["category"] for t in categories]


# Example usage

# Example sentences


def text_list_to_text_comparison(gold, pred, text_sim: Callable[[list, list], list]):
    tp = 0
    n_pos = 0

    sim_matrix = text_sim(gold, pred)
    used = set()
    aligns = []
    for g_i, gold_text in enumerate(gold):
        max_score = 0
        max_ic = None
        for p_i, pred_text in enumerate(pred):
            if p_i not in used:
                s = sim_matrix[g_i][p_i]
                if s > max_score:
                    max_score = s
                    max_ic = p_i, pred_text

        if max_ic is not None:
            # If max score is lower than threshold, consider it as no prediction
            p_i, pred_text = max_ic
            used.add(p_i)
            tp += max_score
            n_pos += 1
            print((gold_text, pred_text))
            aligns.append((gold_text, pred_text))

    w_precision = tp / len(pred)
    w_recall = tp / len(gold)
    f1 = 2 * w_precision * w_recall / (w_precision + w_recall)
    return {"precision": w_precision, "recall": w_recall, "f1": f1, "n_pos": len(pred),
            "aligns": aligns,
            "n_gold": len(gold)}


def main(run_name, do_report=False):
    dataset = "oam"
    save_path: str = get_rule_gen_save_path(run_name, dataset)
    preds = json.load(open(save_path, "r"))
    labels = load_open_ai_category_labels()
    print_metrics = ["f1", "n_pos", "n_gold", "precision", "recall"]
    metrics_to_report = ["f1"]


    text_sim = TextSimilarity()
    score_d = text_list_to_text_comparison(labels, preds, text_sim.compute_similarity_matrix)

    for metric in print_metrics:
        print(f"{metric}\t{score_d[metric]}")

    if do_report:
        proxy = get_task_manager_proxy()
        for metric in metrics_to_report:
            metric_short = metric[:3]
            proxy.report_number(run_name, score_d[metric], dataset, metric_short)


if __name__ == "__main__":
    fire.Fire(main)
