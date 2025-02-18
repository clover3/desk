import random
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def get_random_split_location(tokens) -> Tuple[int, int]:
    retry = True
    n_retry = 0
    while retry:
        st = random.randint(0, len(tokens) - 1)
        while 0 <= st < len(tokens) - 1 and tokens[st].startswith("##"):
            st += 1

        # st is located at end of the text
        if st + 1 > len(tokens) and n_retry < 4:
            n_retry += 1
            retry = True
            continue

        ed = random.randint(st+1, len(tokens))
        retry = False
        return st, ed



def select_random_loc_not_sharp(tokens, st=0, ed=None) -> int:
    if ed is None:
        ed = len(tokens)

    candidate = get_non_sharp_indices(tokens, st, ed)

    if not candidate:
        return -1
    else:
        j = random.randint(0, len(candidate)-1)
        return candidate[j]


def get_non_sharp_indices(tokens, st=0, ed=None):
    if ed is None:
        ed = len(tokens)

    candidate = []
    for i in range(st, ed):
        if tokens[i].startswith("##"):
            pass
        else:
            candidate.append(i)
    return candidate


def get_random_split_location2(tokens) -> Tuple[int, int]:
    retry = True
    n_retry = 0
    while retry:
        st = select_random_loc_not_sharp(tokens, 0, len(tokens))
        # st is located at end of the text
        if st + 1 > len(tokens) and n_retry < 4:
            n_retry += 1
            retry = True
            continue

        ed = select_random_loc_not_sharp(tokens, st+1, len(tokens))
        if ed == -1:
            ed = len(tokens)
        return st, ed



def random_token_split(tokens):
    st, ed = get_random_split_location2(tokens)
    first_a = tokens[:st]
    first_b = tokens[ed:]
    first = first_a + ["[MASK]"] + first_b
    second = tokens[st:ed]
    return first, second