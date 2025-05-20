import json

from rule_gen.reddit.keyword_building.run6.common import load_run6_term_to_text
from rule_gen.reddit.path_helper import get_rp_path


def main():
    score_path = get_rp_path("clustering", f"100.json")
    j = json.load(open(score_path))
    save_path = get_rp_path("clustering", f"100_annot.json")
    d = {}
    for n in range(1, 11):
        d.update(load_run6_term_to_text(n))


    def conv(term):
        try:
            if type(term) == list:
                return d[tuple(term)]
            else:
                return term
        except KeyError as e:
            print(e)
            return " ".join(term)

    out_f = open(save_path, "w")
    out_f.write("[\n")
    for e in j:
        out_f.write("{\n")
        out_f.write("\"cluster_no\": {}\n".format(e["cluster_no"]))
        terms = e["terms"]
        terms_text = list(map(conv, terms))
        st = 0
        step = 10
        while st < 100 and st < len(terms_text):
            term_s = json.dumps(terms_text[st:st+step])
            st += step
            out_f.write("\"terms{}\": {},\n".format(st, term_s))
        out_f.write("\"name\": \"X\"\n")
        out_f.write("},\n")
    out_f.write("]\n")


if __name__ == "__main__":
    main()
