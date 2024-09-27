from toxicity.table_printer.print_util import print_scores_by_asking_server


def main():
    for i in range(0, 10):
        method_list = [
            "lg2_2",
            f"ft12_fold_{i}",
        ]
        method_name_map = None
        condition_list = [
            f"toxigen_train_fold_{i}",
            f"toxigen_train_fail_fold_{i}",
            f"toxigen_train_fail_para_fold_{i}",
            f"toxigen_test_fold_{i}",
        ]
        target_field = "acc"
        print_scores_by_asking_server(
            condition_list,
            method_list,
            target_field,
            method_name_map)


if __name__ == "__main__":
    main()
