from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.reddit.questions.run_question_like import load_reddit_rules_questions
from toxicity.runnable.run_eval_clf import run_eval_clf


def get_api_classifier(question):
    max_text_len = 5000
    from toxicity.apis.open_ai import OpenAIChatClient
    client = OpenAIChatClient("gpt-4o")
    pos_keyword = "yes"
    instruction = question
    instruction += f" If so, output '{pos_keyword}' as a first token. If not, output 'no'"

    def predict(text):
        prompt = instruction + "\n" + text[:max_text_len]
        ret_text = client.request(prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0

    return predict


def main():
    sb = "churning"
    dataset = f"{sb}_val_100"
    run_fmt = "chatgpt_sq_{}"
    questions = load_reddit_rules_questions(sb)
    for q_idx, q in enumerate(questions):
        run_name = run_fmt.format(q_idx)
        predict_fn = get_api_classifier(q)
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        run_eval_clf(run_name, dataset,
                     False, "")


if __name__ == "__main__":
    main()