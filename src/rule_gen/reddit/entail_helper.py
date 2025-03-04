from desk_util.io_helper import read_csv


def load_statement_appy_result(entail_save_path):
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}
    return d


def load_statement_appy_result_as_table(entail_save_path) -> list[list[int]]:
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    d = {}

    for k_idx, t_idx, ret in res:
        k_idx, t_idx = int(k_idx), int(t_idx)
        d.setdefault(k_idx, {})[t_idx] = 1 if ret == "True" else 0

    max_key_idx = max(d.keys())
    n_key = max_key_idx + 1

    result = [[] for _ in range(n_key)]
    key_error_count = 0

    for k_idx, t_dict in d.items():
        max_t_idx = max(t_dict.keys())
        temp_list = []
        for i in range(max_t_idx + 1):
            try:
                temp_list.append(t_dict[i])
            except KeyError:
                temp_list.append(0)
                key_error_count += 1
        result[k_idx] = temp_list

    if key_error_count:
        print(f"Number of KeyError occurrences: {key_error_count}")
    return result
