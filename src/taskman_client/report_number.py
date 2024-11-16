import sys

import fire

from taskman_client.task_proxy import get_task_manager_proxy


def report_number(name, number, condition, field):
    proxy = get_task_manager_proxy()
    proxy.report_number(name, number, condition, field)


if __name__ == "__main__":
    fire.Fire(report_number)
