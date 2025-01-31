from desk_util.io_helper import read_csv


def build_feature_matrix(paired_data: list[tuple[str, str]], entail_save_path):
    data_len = len(paired_data)
    X = build_feature(data_len, entail_save_path)
    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(paired_data)}
    y = []
    for t_idx in range(data_len):
        y.append(label_d[t_idx])
    return X, y


def build_feature(data_len, entail_save_path):
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    print("{} lines".format(len(res)))
    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}
    print("Feature sparsity = {}".format(sum(d.values()) / len(d)))
    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res]) + 1
    print("Use {} keywords".format(max_k_idx))
    X = []
    for t_idx in range(data_len):
        features = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        X.append(features)
    return X
