import logging

from toxicity.table_printer.print_util import print_scores_by_asking_server



def main():

    method_list = [
        "api_1",
        "api_1_t100",
    ]
    method_name_map = {
        "api_1": "Before",
        "api_1_t100": "After",
    }
    condition_list = [
        "toxigen_train_head_100",
        "toxigen_head_100_para_clean",
        "toxigen_test"
    ]
    target_field = "acc"
    print_scores_by_asking_server(
        condition_list, 
        method_list, 
        target_field, 
        method_name_map)


if __name__ == "__main__":
    main()
