import fire

from desk_util.io_helper import save_jsonl
from desk_util.path_helper import get_feature_pred_save_path
from rule_gen.reddit.keyword_building.run3.ask_to_llama import load_feature_pred


def main(sb= "TwoXChromosomes"):
    dataset = f"{sb}_2_train_100"
    run_name = f"llama_rp_cq_{sb}"

    preds = load_feature_pred(run_name, dataset)
    save_path = get_feature_pred_save_path(run_name, dataset)

    key = "</instruction>assistant\n\n"
    for d in preds:
        ret = []
        for s in d["result"]:
            idx = s.find(key)
            if idx == -1:
                raise KeyError
            response = s[idx + len(key):]
            ret.append(int("yes" in response.lower()))
        d["result"] = ret
    save_jsonl(preds, save_path)



def main2(sb= "hearthstone"):
    dataset = f"{sb}_2_train_100"
    run_name = f"llama_rp_cq_{sb}"

    preds = load_feature_pred(run_name, dataset)
    save_path = get_feature_pred_save_path(run_name, dataset)

    for d in preds:
        ret = []
        for s in d["result"]:
            ret.append(int(s))
        d["result"] = ret
    save_jsonl(preds, save_path)


if __name__ == "__main__":
    fire.Fire(main2)