import tqdm
from typing import Callable
from tqdm import tqdm
from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.classifier_loader.get_pipeline import get_classifier_pipeline
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex
import fire



def main(sb= "TwoXChromosomes",):
    role = "train"
    save_dataset_name = f"train_data2_mix"
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", role)
    items = read_csv(train_data_path)
    run_name = "bert2_{}".format(sb)
    predict_fn: Callable[[str], tuple[str, float]] = get_classifier_pipeline(run_name)
    save_path = get_clf_pred_save_path(run_name, save_dataset_name)

    def predict(e):
        text, _label = e
        label, score = predict_fn(text)
        return "noid", label, score

    pred_itr = map(predict, tqdm(items, desc="Processing", unit="item"))
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")



if __name__ == "__main__":
    fire.Fire(main)