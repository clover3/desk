from collections import defaultdict

import fire

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.io_helper import load_jsonl
from desk_util.path_helper import get_feature_pred_save_path
from rule_gen.reddit.keyword_building.run3.ask_to_llama import load_q_lists


def parse_per_feature_outputs(j_list, q_list) -> dict[int, list[tuple[int, bool, float]]]:
    per_q = defaultdict(list)
    for i, out_d in enumerate(j_list):
        for q_i, q in enumerate(q_list):
            pred, score = out_d["result"][q_i]
            if not pred:
                score = -score
            per_q[q_i].append((i, pred, score))
    return per_q


def main(sb="fantasyfootball"):
    dataset = f"{sb}_2_val_100"
    run_name = f"llama2_rp_cpq2_{sb}2"
    payload = load_csv_dataset_by_name(dataset)
    save_path = get_feature_pred_save_path(run_name, dataset)
    j_list = load_jsonl(save_path)
    tokens = run_name.split("_")
    engine_name = tokens[0]
    text_name = tokens[1]
    short_name = tokens[2]
    sb = tokens[3]
    q_list = load_q_lists(text_name, short_name, sb)
    text_list: list[str] = right(payload)

    per_q = parse_per_feature_outputs(j_list, q_list)

    for q_i, q in enumerate(q_list):
        print("----")
        print(q)

        entries = per_q[q_i]
        entries.sort(key=lambda x: x[2], reverse=True)
        table = []
        for i, pred, score in entries:
            text = text_list[i]
            text = text.replace("\n", "\\n")
            text = text[:200] + " ..." if len(text) > 200 else text
            row = [pred, score, text]
            table.append(row)
        print_table(table)


if __name__ == "__main__":
    fire.Fire(main)
