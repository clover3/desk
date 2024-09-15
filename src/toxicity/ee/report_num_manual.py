from taskman_client.task_proxy import get_task_manager_proxy


def main():
    proxy = get_task_manager_proxy()
    run_name = "grace_l28"
    post_metrics = {'train_acc': 0.84, 'train_auc': 0.9117276166456494, 'test_acc': 0.81, 'test_auc': 0.8990147783251232, 'para_acc': 0.83, 'para_auc': 0.8970155527532576}
    todo = {
        "train_acc": "toxigen_train_head_100",
        "test_acc": "toxigen_test_head_100",
        "para_acc": "toxigen_head_100_para_clean",
    }
    for key, long_name in todo.items():
        proxy.report_number(run_name, float(post_metrics[key]), long_name, "acc")


if __name__ == "__main__":
    main()