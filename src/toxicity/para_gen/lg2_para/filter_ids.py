from desk_util.io_helper import read_csv_column, save_text_list_as_csv
from desk_util.path_helper import get_wrong_pred_save_path, \
    get_text_list_save_path


def main():
    save_path: str = get_wrong_pred_save_path("lg2_2", "toxigen_train_fold_all")
    ids_todo = read_csv_column(save_path, 0)
    save_path: str = get_wrong_pred_save_path("some_run", "toxigen_train_fold_all")
    ids_have = read_csv_column(save_path, 0)
    ids_remain = [t for t in ids_todo if t not in ids_have]
    print("{} items".format(len(ids_remain)))

    s = get_text_list_save_path("ids_to_gen")
    save_text_list_as_csv(ids_remain, s)


if __name__ == "__main__":
    main()
