import json
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class NGramInfo:
    sub_text: str
    text: str
    score: float


def load_ngram_outputs(path):
    j = json.load(open(path))

    def parse(items: list[list[dict]]) -> list[list[NGramInfo]]:
        results = []
        for item_list in items:
            ngram_list = []
            for d in item_list:
                ngram_list.append(NGramInfo(**d))
            results.append(ngram_list)

        return results

    pos = parse(j['pos'])
    neg = parse(j['neg'])
    return pos, neg
