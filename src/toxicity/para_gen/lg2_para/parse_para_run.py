from toxicity.io_helper import save_text_list_as_csv, read_csv_column
from toxicity.para_gen.ft12_first.parse_para_by_rule import parse_para
from toxicity.path_helper import get_text_list_save_path


def main():
    save_path = get_text_list_save_path(f"para_from_ids_to_gen")
    text_itr = read_csv_column(save_path, 0)
    output = parse_para(text_itr)
    save_path = get_text_list_save_path(f"toxigen_train_para_all_fold_selected_v2")
    save_text_list_as_csv(output, save_path)




if __name__ == "__main__":
    main()