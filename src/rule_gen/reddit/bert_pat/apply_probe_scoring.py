import os
import json
import fire
from tqdm import tqdm
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_probe.probe_inference import ProbeInference
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def apply_probe_scoring(sb="TwoXChromosomes"):
    model_name = f"bert2_{sb}"
    model_name = model_name + "_probe"
    model_path = get_model_save_path(model_name)
    role = "train"

    save_path = get_reddit_train_data_path_ex("train_data2", sb, role)
    items = read_csv(save_path)
    max_length = 256
    inf = ProbeInference(model_path)
    probe_save_path = os.path.join(
        output_root_path, "reddit", f"train_data2_probe_{role}", f"{sb}.txt")

    with open(probe_save_path, "w", encoding="utf-8") as f:
        for text, label in tqdm(items):
            probe_score = inf.predict_get_layer_mean(text)
            probe_score = probe_score[0, :, 0].tolist()
            probe_score = [round(t, 2) for t in probe_score]
            inputs = inf.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
            )
            tokens = inf.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
            j = {
                'text': text,
                'label': label,
                'tokens': tokens,
                "attention_mask": inputs["attention_mask"],
                'probe_score': probe_score,
            }
            f.write(json.dumps(j) + "\n")



if __name__ == "__main__":
    fire.Fire(apply_probe_scoring)
