from desk_util.print_util import print_scores_by_asking_server


def main():
    method_list = [
        "NONE",
        "lgt2_mlp_28_e20",
        "lgt2_mlp_28_e10",
        "lgt2_mlp_28",
        "lora28",
        "ee_ft_eow_l28",
        "grace28",
        "wise_l24",
        "ee_wise_l21",
    ]
    method_name_map = None
    condition_list = [
        "toxigen_train_head_100",
        "toxigen_test_head_100",
        "toxigen_head_100_para_clean"
    ]
    target_field = "acc"
    print_scores_by_asking_server(
        condition_list,
        method_list,
        target_field,
        method_name_map)


if __name__ == "__main__":
    main()
