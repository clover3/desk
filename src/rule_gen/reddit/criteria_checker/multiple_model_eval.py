from collections import Counter

from chair.misc_lib import Averager
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.cpath import output_root_path
import os
import json


def main():
    subreddit_list = get_split_subreddit_list("train")
    # save_path = os.path.join(output_root_path, "reddit", "mod_sents.json")
    # sents = json.load(open(save_path))
    save_path = os.path.join(output_root_path, "reddit", "subset", "mod_neg.json")
    sents = json.load(open(save_path))
    n_sb = 60

    counter = Counter()
    predict_fn_d = {}
    for s in sents:
        for i, sb in enumerate(subreddit_list[:n_sb]):
            run_name = "bert2_{}".format(sb)
            if run_name not in predict_fn_d:
                print("Loading {}".format(run_name))
                predict_fn = get_classifier(run_name)
                predict_fn_d[run_name] = predict_fn
            else:
                predict_fn = predict_fn_d[run_name]

            pred, _score = predict_fn(s)
            pred = int(pred)
            counter[sb, pred] += 1


    all_pos_cnt = 0
    averager = Averager()
    for sb in subreddit_list:
        rate = counter[sb, 1] / (counter[sb, 0] + counter[sb, 1])
        averager.append(rate)
        print("{}: {} {}".format(sb, counter[sb, 1], counter[sb, 0]))
        if counter[sb, 0]  == 0:
            all_pos_cnt += 1

    print("Average positive rate", averager.get_average())
    print("All positive cnt", all_pos_cnt)


if __name__ == "__main__":
    main()

