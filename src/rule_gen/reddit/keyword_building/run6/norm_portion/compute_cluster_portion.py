import json
import pickle
from collections import Counter
from chair.misc_lib import Averager
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_terms
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms
from rule_gen.reddit.path_helper import load_subreddit_list, get_rp_path
from chair.tab_print import print_table


def get_term_to_cluster():
    score_path = get_rp_path("clustering", f"100.json")
    j = json.load(open(score_path, "r"))
    term_to_cluster_id = {}
    skip_n = 0
    for e in j:
        no = e["cluster_no"]
        # if e["average"] < 0.5:
        #     skip_n += 1
        #     continue
        for t in e["terms"]:
            term_to_cluster_id[tuple(t)] = no
    print("Skip {} clusters".format(skip_n))
    return term_to_cluster_id

def cluster_topic():
    annot_path = get_rp_path("clustering", f"100_annot.json")
    j = json.load(open(annot_path))
    d = {}
    for e in j:
        no = e["cluster_no"]
        d[no] = e['name']
    return d


def main():
    # Initialize voca -> cluster mapping
    # Load all voca.
    sb_list = load_subreddit_list()
    cluster_map: dict[tuple, int] = get_term_to_cluster()
    c_topics: dict[int, str] = cluster_topic()
    n_list = list(range(1, 10))
    score_mat, term_list, valid_sb_list = load_mat_terms(n_list)

    voca = set()
    for n in n_list:
        terms = load_run6_10k_terms(n)
        if n == 1:
            assert type(terms[0]) == str
            terms = [(t,) for t in terms]
        voca.update(terms)

    def subset(pattern, long):
        for k in pattern:
            if k not in long:
                return False
        return True


    n_docs = 100
    tf_list = []
    for sb in sb_list:
        try:
            sb_idx = valid_sb_list.index(sb)
        except ValueError:
            continue

        tf = Counter()
        doc_term_match_path = get_rp_path("doc_term_match", f"{sb}.json")
        doc_terms = json.load(open(doc_term_match_path, "r"))
        g_count = Counter()
        for doc_id, terms in doc_terms[:n_docs]:
            portion = Counter()
            terms.sort(key=lambda x: len(x), reverse=True)
            seen_term = set()
            for t in terms:
                t = tuple(t)
                if len(t) == 1:
                    idx = term_list.index(t[0])
                else:
                    idx = term_list.index(t)

                score = score_mat[idx, sb_idx]
                if score < 0.8:
                    continue
                if t in cluster_map:
                    portion[cluster_map[t]] += 1
                    seen_term.add(t)
                else:
                    matched = False
                    for k in seen_term:
                        if subset(t, k):
                            matched = True
                            break
                    if not matched:
                        tf[t] += 1
                        portion["unknown"] += 1


            if len(portion) == 0:
                unknown_portion = 1
                g_count["no match"] += 1
            else:
                s = sum(portion.values())
                for k, v in portion.items():
                    # g_count[k] += v / s
                    g_count[k] += v

        print("---------")
        print(sb)
        for k, v in g_count.most_common(50):
            cluster_rep = c_topics[k] if k in c_topics else k
            print("{} ({:.2f})".format(cluster_rep, v))
        tf_list.append((sb, tf))

    # doc_term_match_path = get_rp_path("clustering", f"100_annot_missing.pkl")
    # pickle.dump(tf_list, open(doc_term_match_path, "wb"))
        # table.append((sb, averager.get_average()))
    # table.sort(key=lambda x: x[1], reverse=True)
    # print_table(table)


if __name__ == "__main__":
    main()
