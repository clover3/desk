
from toxicity.table_printer.print_util import print_scores_by_asking_server



def main():

    method_list = []
    method_name_map = {}
    for i in range(1, 33):
        name = f"ee_ft_l{i}"
        method_list.append(name)
        method_name_map[name] = str(i)

    condition_list = [
        "toxigen_train_head_100",
        "toxigen_test_head_100",
    ]
    target_field = "acc"
    print_scores_by_asking_server(
        condition_list,
        method_list,
        target_field,
        method_name_map)


if __name__ == "__main__":
    main()
