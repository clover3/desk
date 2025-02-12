
{common} = src/rule_gen/reddit
* Steps
  * Train a classifier
    * {common}/base_bert
  * Train a probe model on the classifier
    * {common}/bert_probe
  * Extract embeddings of texts using the classifier
    * {common}/base_bert/extract_embeddings.py
  * Cluster the texts based on embeddings 
  * Select tokens from clusters' texts using the probe model
  * Query GPT what is common in selected parts
    * Get First response
      * {common}/build_rule/cluster_probe.py
    * Change response to a question format
      * {common}/keyword_building/run3/extract_questions.py
    * Parse response into json
      * 
