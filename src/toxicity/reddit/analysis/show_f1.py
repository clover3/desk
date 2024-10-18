import os

from taskman_client.named_number_proxy import NamedNumberProxy
from toxicity.cpath import output_root_path
from toxicity.io_helper import read_csv, save_csv
from toxicity.reddit.path_helper import get_split_subreddit_list, get_reddit_training_data_size, \
    get_reddit_train_data_path, load_subreddit_list
import scipy.stats



def main():
    subreddit_list = get_split_subreddit_list("train")
    search = NamedNumberProxy()
    data_size = get_reddit_training_data_size()

    output = []
    for sb in subreddit_list:
        model_name = f"bert_{sb}"
        ret = search.get_number(model_name, "f1")
        n = data_size[sb]
        row = [model_name, n, ret]
        print(row)
        output.append(row)

    save_path = os.path.join(output_root_path, "reddit", "bert_f1.csv")
    save_csv(output, save_path)




def show_f1_api_runs():
    subreddit_list = get_split_subreddit_list("train")
    search = NamedNumberProxy()

    output = []
    for sb in subreddit_list:
        model_name = f"api_{sb}_detail"
        ret = search.get_number(model_name, "f1")
        row = [model_name, ret]
        print(row)
        output.append(row)

    save_path = os.path.join(output_root_path, "reddit", "api_f1.csv")
    save_csv(output, save_path)

def f1_corr():
    save_path = os.path.join(output_root_path, "reddit", "bert_f1.csv")
    items = read_csv(save_path)
    f1s = []
    n_s = []
    for subreddit, n, f1 in items:
        try:
            f1 = float(f1)
            n = int(n)
            f1s.append(f1)
            n_s.append(n)
        except ValueError:
            pass

    # PearsonRResult(statistic=-0.03446086910611207, pvalue=0.7955485428256622)
    print(scipy.stats.pearsonr(f1s, n_s))



if __name__ == "__main__":
    show_f1_api_runs()
