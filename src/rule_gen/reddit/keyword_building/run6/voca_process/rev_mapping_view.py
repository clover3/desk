import pickle

from rule_gen.reddit.path_helper import get_rp_path


def main():
    rev_path = get_rp_path("run6_voca_rev_src_map.pkl")
    rev_map: dict[int, dict[str, str]] = pickle.load(open(rev_path, "rb"))
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))

    for n in range(2, 10):
        voca = voca_d[n]
        print("Voca len", len(voca))
        print("rev len", len(rev_map[n]))
        assert type(voca) is list


if __name__ == "__main__":
    main()
