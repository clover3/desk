import json
import random
import fire

from rule_gen.reddit.path_helper import get_rp_path


def load_pattern_list_from_clusters(clusters, sub_to_text):
    pattern_list = []
    for cluster in clusters:
        text_list = cluster["text_list"]
        random.shuffle(text_list)
        for sub_text in text_list:
            sub_text = sub_text.strip()
            if sub_text in sub_to_text:
                text = sub_to_text[sub_text]
                st = text.find(sub_text)
                if st == -1:
                    continue

                pattern_list.append((text, sub_text))
                break
    return pattern_list


def main(sb="TwoXChromosomes"):
    print(sb)
    subtext_path = get_rp_path("ngram_93_all_sub_sel", f"{sb}.json")
    sub_texts = json.load(open(subtext_path))
    sub_to_text = {}
    for e in sub_texts:
        sub_to_text[e["sub_text"]] = e["text"]

    cluster_path = get_rp_path("ngram_93_all_sel_cluster", f"{sb}.json")
    try:
        clusters = json.load(open(cluster_path))
        pattern_list = load_pattern_list_from_clusters(clusters, sub_to_text)
        print("Load pattern from clusters")
    except FileNotFoundError:
        sub_text_path = get_rp_path("ngram_93_all_sub_sel", f"{sb}.json")
        sub_text = json.load(open(sub_text_path))
        pattern_list = []
        for e in sub_text:
            pattern_list.append((e["text"], e["sub_text"]))
        print("Load pattern from ngram_93_all_sub_sel")

    print("Got %d patterns" % len(pattern_list))
    context_len = 100
    rule_list = []
    for text, sub_text in pattern_list:
        st = text.find(sub_text)
        pre_context = text[:st]
        post_context = text[st + len(sub_text):]
        if len(pre_context) > context_len:
            pre_context = " ... " + pre_context[-context_len:]
        if len(post_context) > context_len:
            post_context = post_context[:context_len] + " ... "

        rule_str = f"{pre_context}<reason>{sub_text}</reason>{post_context}"
        rule_list.append(rule_str)

    rule_prompt_path = get_rp_path("ngram_93_rule1", f"{sb}.json")

    with open(rule_prompt_path, 'w') as f:
        json.dump(rule_list, f, indent=2)



if __name__ == "__main__":
    fire.Fire(main)