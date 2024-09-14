import sys

import fire

from taskman_client.task_proxy import get_task_manager_proxy
import numpy as np


def main():
    proxy = get_task_manager_proxy()
    number = np.mean([0.1, 0.2])
    proxy.report_number("test", number, condition="", field="")


if __name__ == "__main__":
    main()
