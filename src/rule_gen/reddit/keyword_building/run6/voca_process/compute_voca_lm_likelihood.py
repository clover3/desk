import pickle

import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_user.llama_helper.llama_model_names import Llama3_8B
from rule_gen.reddit.keyword_building.run6.voca_process.compute_lm_prob_dev import compute_text_log_prob
from rule_gen.reddit.path_helper import get_rp_path
import os

def main():
    rev_path = get_rp_path("run6_voca_rev_src_map.pkl")
    rev_map: dict[int, dict[str, str]] = pickle.load(open(rev_path, "rb"))
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))

    model_name = Llama3_8B
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for n in range(1, 11):
        score_path = get_rp_path( "run6_voca_lm_prob", f"{n}.pkl")
        if os.path.exists(score_path):
            print(f"score file {score_path} already exists, skipping.")
            continue

        voca = voca_d[n]
        print("Voca len", len(voca))
        print("rev len", len(rev_map[n]))
        assert type(voca) is list
        out_l = []
        for v in tqdm(voca):
            if n == 1:
                text = v
            else:
                if v in rev_map[n]:
                    text = rev_map[n][v]
                else:
                    text = " ".join(v)
            log_prob = compute_text_log_prob(model, tokenizer, device, text)
            e = v, text, log_prob
            out_l.append(e)

        pickle.dump(out_l, open(score_path, "wb"))


if __name__ == "__main__":
    fire.Fire(main)
