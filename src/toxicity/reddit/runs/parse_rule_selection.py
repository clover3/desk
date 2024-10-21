import re
from collections import Counter

from toxicity.io_helper import read_csv
from toxicity.path_helper import get_dataset_pred_save_path
from toxicity.runnable.run_eval import load_labels


def find_first_number(input_string):
    pattern = r'\b(?:0*(?:[0-9]|[1-9][0-9]))\b|\d{1,2}'
    matches = re.findall(pattern, input_string)
    return int(matches[0]) if matches else None


def main():
    # sb = "TwoXChromosomes"
    sb = "churning"
    print(sb)
    run_name = f"rs_{sb}"
    dataset = f"{sb}_val_100"
    save_path: str = get_dataset_pred_save_path(run_name, dataset)
    labels = load_labels(dataset)
    labels_d = dict(labels)

    data = read_csv(save_path)
    print(len(data), "items")
    counter = Counter()
    for data_id, text, _ in data:
        # if labels_d[data_id]:
        num = find_first_number(text)
        if num is not None:
            counter[num] += 1
        else:
            print(num)

    print(counter)
    print(sum(counter.values()))
    n = max(counter.keys())
    for i in range(1, n+1):
        print(f"{i}\t{counter[i]}")


if __name__ == "__main__":
    main()
