
common = src/rule_gen/reddit/keyword_building

* Step 1
  * {common}/gpt_norms/response_norm_match.py
  * input: norm_voca/man.txt
  * output: norm_voca/doc_id_bow_man_norm.pkl
* Step 2
  * {common}/gpt_norms/pat_term_to_norm_term.py
  * input: norm_voca/doc_id_bow_man_norm.pkl
  * output: run6_voca_to_man_norm/{n}.pkl
* Step 3
  * {common}/gpt_norms/per_sb_norm_bow.py
  * {common}/run6_man_norm_diff