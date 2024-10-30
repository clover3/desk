import fire

from chair.tab_print import print_table
from toxicity.clf_util import eval_prec_recall_f1_acc
from toxicity.io_helper import read_csv
from toxicity.path_helper import get_clf_pred_save_path
from toxicity.reddit.path_helper import get_split_subreddit_list
from toxicity.runnable.run_eval import load_labels, clf_eval, align_preds_and_labels


def main():
    print(__name__)
    subreddit_list = get_split_subreddit_list("train")
    run_name = f"bert_train_mix2"

    output = []
    columns = ["f1", "precision", "recall"]
    head = [""] + columns
    output.append(head)
    al_preds_all = []
    al_labels_all = []
    al_scores_all = []

    for sb in subreddit_list:
        dataset = f"{sb}_val_100"
        try:
            save_path: str = get_clf_pred_save_path(run_name, dataset)
            raw_preds = read_csv(save_path)
            preds = [(data_id, int(pred), float(score)) for data_id, pred, score in raw_preds]
            labels = load_labels(dataset)
            aligned_preds, aligned_labels, aligned_scores = align_preds_and_labels(preds, labels)
            al_preds_all.extend(aligned_preds)
            al_labels_all.extend(aligned_labels)
            al_scores_all.extend(aligned_scores)
        except FileNotFoundError:
            pass

    score_d = eval_prec_recall_f1_acc(al_labels_all, al_preds_all)
    print(score_d)



if __name__ == "__main__":
    main()