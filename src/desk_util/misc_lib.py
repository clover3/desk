import os
from typing import Iterable, TypeVar, Callable, Dict, List

A = TypeVar('A')
B = TypeVar('B')


def exist_or_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def make_parent_exists(target_path):
    def make_dir_parent_exists(target_dir):
        parent_path = os.path.dirname(target_dir)
        if not os.path.exists(parent_path):
            make_dir_parent_exists(parent_path)
        exist_or_mkdir(target_dir)

    parent_path = os.path.dirname(target_path)
    make_dir_parent_exists(parent_path)


def get_second(x):
    return x[1]


def group_by(interable: Iterable[A], key_fn: Callable[[A], B]) -> Dict[B, List[A]]:
    grouped = {}
    for elem in interable:
        key = key_fn(elem)
        if key not in grouped:
            grouped[key] = list()

        grouped[key].append(elem)
    return grouped
