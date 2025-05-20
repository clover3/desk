import numpy as np
from chair.tab_print import print_table
from rule_gen.reddit.keyword_building.run6.common import load_run6_term_text_to_term
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms_pickled

words = [
    "whore",
    "pussy",
    "trump2016",
    "homophobia",
    "muslims",
    "lesbians",
    "jews",
    "bitches",
    "racist",
    "lesbian",
    "nigger",
    "idiot",
    "trump",
    "lesbianism",
    "pathetic",
    "bitch",
    "motherfucker",
]


words = ["russian",
 "do you speak english",
 "the story of andrew jackson",
 "tesla",
 "holy shit",
 "asian man",
 "fucked up",
 "trump",
 ]

{"russian": "accuse each other as russian spy"}
def main():
    score_mat, term_list, valid_sb_list = load_mat_terms_pickled()
    mean_scores = np.mean(score_mat, axis=1)

    sb_list = ["politics", "Games", "history"]
    head = ["", "mean"] + sb_list
    sb_i_list = [valid_sb_list.index(sb) for sb in sb_list]
    table = [head]
    for t in words:
        tokens = t.split()
        if len(tokens) > 1:
            t = tuple(tokens)
        i = term_list.index(t)
        row = [t]
        row.append("{:.2f}".format(mean_scores[i]))
        for sb_i in sb_i_list:
            row.append("{:.2f}".format(score_mat[i, sb_i]))
        table.append(row)
    print_table(table)



if __name__ == "__main__":
    main()