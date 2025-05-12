import os
import pickle
from collections import defaultdict

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_rp_path


def main():
    voca_path = get_rp_path( "run6_voca.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))
    oud_voca_d = defaultdict(list)
    for n, may_set in voca_d.items():
        oud_voca_d[n].extend(may_set)

    voca_l_path = get_rp_path( "run6_voca_l.pkl")
    pickle.dump(oud_voca_d, open(voca_l_path, "wb"))

    for n in range(10):
        voca_l_path = get_rp_path(
                                   "run6_voca_l", f"{n}.txt")
        make_parent_exists(voca_l_path)
        with open(voca_l_path, "w") as f:
            for term in oud_voca_d[n]:
                f.write(f"{term}\n")


if __name__ == "__main__":
    main()
