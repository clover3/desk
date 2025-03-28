import json
import logging
LOG = logging.getLogger(__name__)


def show_payload():
    j_path = "/home/qdb5sn/work/LLaMA-Factory/data/lf_pattern_h.json"
    j = json.load(open(j_path))
    for t in j[:50]:
        print("----")
        print(t['input'])
        print(t['output'])


if __name__ == "__main__":
    show_payload()
