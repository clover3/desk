import json
import os
from collections import Counter

from chair.list_lib import right, left
from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from desk_util.path_helper import load_csv_dataset
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex
from rule_gen.reddit.s9.feature_extractor import extract_ngram_features, get_value
from rule_gen.cpath import output_root_path
import os

def main():

    trivial_feature_save_path = ""
    for split in ["train", "val"]:
        todo = get_split_subreddit_list(split)
        matched_sub = []
        for sb in todo:
            out_save_path = os.path.join(
                output_root_path, "reddit",
                "rule_processing", "trivial_features", f"{sb}.json")

            min_df = 20
            data = read_csv(get_reddit_train_data_path_ex("train_data2", sb, "train"))
            data = data[:1000]
            text_list = left(data)
            y_true = list(map(int, right(data)))
            def extract_fn(text):
                return extract_ngram_features(text, 3, 7)
            feature_dict_list: list[dict[tuple, int]] = list(map(extract_fn, text_list))
            df = Counter()
            for d in feature_dict_list:
                for key in d:
                    df[key] += 1

            def select_features(feature_dict_list, y_true):
                table = []
                for f in selected_features:
                    X: list[int] = [t[f] for t in feature_dict_list]
                    X = [1 if x > 0 else 0 for x in X]
                    prec, support = get_value(y_true, X)
                    if prec == 1 and support > 4:
                        row = [f, prec, support]
                        table.append(row)
                return table
            selected_features = {k for k, v in df.items() if v >= min_df}
            n_try = 0
            save_content = []
            while n_try < 5:
                table = select_features(feature_dict_list, y_true)
                if table:
                    if sb not in matched_sub:
                        matched_sub.append(sb)

                    def feature_priority(e):
                        key1 = e[2]
                        key2 = len(e[0])
                        return key1, key2
                    table.sort(key=feature_priority, reverse=True)
                    top_feature = table[0][0]
                    save_content.append(" ".join(top_feature))
                    y_old, x_old = y_true, feature_dict_list
                    new_x = []
                    new_y = []
                    for x, y in zip(x_old, y_old):
                        if top_feature not in x:
                            new_x.append(x)
                            new_y.append(y)

                    if len(y_true) == len(new_y):
                        raise ValueError()
                    y_true = new_y
                    feature_dict_list = new_x
                    n_try += 1
                else:
                    break
            if save_content:
                json.dump(save_content, open(out_save_path, "w"), indent=4)

        print("Detected subs: ", matched_sub)



if __name__ == "__main__":
    main()