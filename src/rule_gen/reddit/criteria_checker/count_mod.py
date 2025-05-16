import os
from collections import Counter

from chair.misc_lib import group_by
from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_analysis.common import load_run_score_matrix


def get_term_score(term):
    dir_name = "run6_10k_score"
    n = 1
    score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)
    term_list = load_run6_10k_text(n)
    idx = term_list.index(term)
    return score_mat[idx], valid_sb_list



def main():
    text_path = os.path.join(output_root_path, "reddit", "subset", "mod.csv")
    data = read_csv(text_path)
    scores, sb_list = get_term_score("mod")
    table_scores = dict(zip(sb_list, scores))
    groups = group_by(data, lambda x: x[0])



    table = []
    for sb, entries in groups.items():
        counter = Counter()
        for _sb, _text, label_s in entries:
            counter[int(label_s)] += 1

        try:
            if len(counter) > 0:
                total = sum(counter.values())
                table.append((sb, counter[1]/total, table_scores[sb]))
            else:
                table.append((sb, -1, table_scores[sb]))
        except KeyError as e:
            print(e)

    table.sort(key=lambda x: x[2], reverse=True)
    for sb, portion, score in table:
        print(sb, portion, score)


if __name__ == "__main__":
    main()
