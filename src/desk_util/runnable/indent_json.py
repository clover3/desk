import json
import os
import fire
from rule_gen.cpath import output_root_path


def indent_and_print(path):
    obj = json.load(open(path, "r"))
    s = json.dumps(obj, indent=4, ensure_ascii=False)
    print(s)



if __name__ == "__main__":
    fire.Fire(indent_and_print)
