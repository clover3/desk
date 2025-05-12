import os
import pickle
import fire
import tqdm
from chair.misc_lib import make_parent_exists
from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
from rule_gen.reddit.bert_pat.infer_tokens import PatInferenceFirst
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main(n=1):
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))
    voca: list = voca_d[n]
    assert type(voca) == list
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        model_name = f"bert_ts_{sb}"
        pat = PatInferenceFirst(get_model_save_path(model_name))
        save_path = get_rp_path("run6_voca", f"{sb}.{n}.pkl")
        if os.path.exists(save_path):
            print(save_path, "already not exist")
            continue

        make_parent_exists(save_path)
        data_itr = tqdm.tqdm(voca)
        scores = []
        for t in data_itr:
            if n == 1:
                text = t
            else:
                text = " ".join(t)
            scores.append(pat.get_full_text_score(text))
        pickle.dump(scores, open(save_path, "wb"))


if __name__ == "__main__":
    fire.Fire(main)
