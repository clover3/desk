import json
from dataclasses import dataclass
from typing import List, Tuple

from chair.misc_lib import group_by


@dataclass
class NGramInfo:
    sub_text: str
    text: str
    score: float


@dataclass
class NGramInfo2:
    text: str
    sub_text_list: list[tuple[str, float]]


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


def load_ngram_outputs_grouped(path):
    pos, neg = load_ngram_outputs(path)

    def apply_group(l: list[list[NGramInfo]]):
        output = []
        for entries in l:
            if not entries:
                continue
            key = entries[0].text
            subs = [(e.sub_text, e.score) for e in entries]
            output.append({"text": key, "subs": subs})
        return output

    return {
        "pos": apply_group(pos),
        "neg": apply_group(neg)
    }
