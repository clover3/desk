

src/rule_gen/reddit/base_bert/extract_embeddings.py
* Model: bert2_{sb}
* Input: train_data2/{sb}/train
* Output: reddit/pickles/bert2_{sb}.pkl
* Comment: Extract embedding as [CLS] Token pooling


src/rule_gen/reddit/base_bert/clusterings.py
* Input: reddit/pickles/bert2_{sb}.pkl
* Output: clusters/bert2_{sb}.json
* 

src/rule_gen/reddit/bert_probe/probe_sel_top_clusters.py
* Model: bert2_{sb}_probe
* Input: clusters/bert2_{sb}.json
* Output: clusters_important

src/rule_gen/reddit/build_rule/cluster_probe.py
* Input: clusters_important
* Output: cluster_probe_prompt
*

src/rule_gen/reddit/keyword_building/run3/extract_question_cluste_probe.py
* Input: cluster_probe_prompt
* Output: cluster_probe_questions
* Last code update: (Feb 11)

src/rule_gen/reddit/keyword_building/run3/ask_to_llama.py
* Run Llama inference
* Input: cluster_probe_questions
* Output: clf/{sb}/cq_{sb}_{i}

cq: cluster_questions
cpq: cluster_probe_questions

src/rule_gen/reddit/rule_classifier/run3/cq_gnq_train_only.py
