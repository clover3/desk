import csv
import json
import logging
import sys
from typing import List, Tuple

from desk_util.misc_lib import make_parent_exists


def save_csv(tuple_itr, file_path: str) -> None:
    make_parent_exists(file_path)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in tuple_itr:
            writer.writerow(row)


def read_csv(file_path: str, return_itr=False):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def read_csv_column(file_path: str, column_i):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        output = []
        for row in csv.reader(f):
            output.append(row[column_i])
        return output


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


load_jsonl = read_jsonl


class IgnoreFilter(logging.Filter):
    def __init__(self, ignore_rules: List[Tuple[str, str]] = None):
        """
        Initialize filter with ignore rules.

        Args:
            ignore_rules: List of tuples (logger_name, string_contains)
                         Both elements can be empty strings to match everything
        """
        super().__init__()
        self.ignore_rules = ignore_rules or []

    def filter(self, record):
        # Return False to ignore the log entry
        for logger_name, contains_str in self.ignore_rules:
            if (not logger_name or record.name.startswith(logger_name)) and \
                    (not contains_str or contains_str in record.getMessage()):
                return False
        return True


f_init = False


def init_logging(level=logging.INFO, ignore_rules: List[Tuple[str, str]] = None):
    global f_init
    if f_init:
        return
    format_str = '%(levelname)s %(name)s %(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    # Add the ignore filter
    ignore_filter = IgnoreFilter(ignore_rules)
    ch.addFilter(ignore_filter)

    root_logger = logging.getLogger()
    root_logger.addHandler(ch)
    root_logger.setLevel(level)
    f_init = True
