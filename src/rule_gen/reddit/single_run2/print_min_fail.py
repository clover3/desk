import json
import logging
from collections import Counter
from rule_gen.cpath import output_root_path
import os
import fire
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.infer_tokens import PatInferenceFirst
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from taskman_client.wrapper3 import JobContext

LOG = logging.getLogger("InfPat")




def main(sb="TwoXChromosomes"):
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"
    LOG.setLevel(logging.INFO)
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_sel", "", f"{sb}.csv")
    make_parent_exists(res_save_path)

    t = 0.8
    n_item = 100
    confusion = Counter()
    output = []
    with JobContext(model_name + "_train"):
        train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
            data_name, sb, "train"))
        train_dataset = train_dataset.take(n_item)
        bert_tms = get_classifier("bert2_train_mix_sel")

        model_path = get_model_save_path(model_name)
        pat = PatInferenceFirst(model_path)
        for i, example in enumerate(train_dataset):
            text = example['text']
            tms_pred, tms_score = bert_tms(text)
            full_score = pat.get_full_text_score(text)
            domain_pred = int(full_score > 0.5)
            confusion[(int(tms_pred), domain_pred)] += 1
            print(text)
            print("---")
            output.append({
                "idx": i,
                "text": text,
                "tms_pred": tms_pred,
                "domain_pred": domain_pred
            })

    json.dump(output, open(res_save_path, "w"), indent=2)


if __name__ == "__main__":
    fire.Fire(main)
