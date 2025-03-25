


* src/rule_gen/reddit/bert_pat/runs/collect_ngram2.py
  * Select strong n-grams
  * Output: ngram_93_all
* src/rule_gen/reddit/bert_pat/feature_sel/select_sub_text.py
  * Input: ngram_93_all
  * output: ngram_93_all_sub_sel:
  * For each text, select one sub text. (High scoring + longer text)
* src/rule_gen/reddit/bert_pat/feature_sel/cluster_pat.py
  * KMeans clustering
  * Input: ngram_93_all_sel:
  * Output: ngram_93_all_sel_cluster
* src/rule_gen/reddit/llama/single_run/gen_f.py
* src/rule_gen/reddit/llama/gen_lf_yaml.py