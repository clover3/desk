import os
import json
from toxicity.cpath import output_root_path
import fire
import logging
from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.reddit.classifier_loader.load_by_name import get_classifier
from toxicity.runnable.run_eval_clf import run_eval_clf


def load_reddit_rules_questions(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rules_re", f"{sb}.json")
    j = json.load(open(rule_save_path, "r"))
    rules = j["rules"]

    questions = []
    for r in rules:
        questions.extend(r["questions"])
    return questions


def get_api_classifier(question):
    max_text_len = 5000
    from toxicity.llama_helper.lf_client import LLMClient
    client = LLMClient(max_prompt_len=5000)
    pos_keyword = "yes"
    prompt = question
    prompt += f" If so, output '{pos_keyword}' as a first token. If not, output 'no'"

    def predict(text):
        text = text[:max_text_len]
        ret_text = client.ask(text, prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def main():
    sb = "churning"
    dataset = f"{sb}_val_100"
    run_fmt = "api_sq_{}"
    questions = load_reddit_rules_questions(sb)
    for q_idx, q in enumerate(questions):
        run_name = run_fmt.format(q_idx)
        predict_fn = get_api_classifier(q)
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        run_eval_clf(run_name, dataset,
                     False, "")


if __name__ == "__main__":
    main()