import fire

from toxicity.apis.open_ai_batch_requester import BatchChatGPTLoader
from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.reddit.classifier_loader.load_by_name import get_instruction_from_run_name, PromptBuilder


def get_offline_clf_predictor(run_name, dataset):
    p_builder = PromptBuilder(run_name)
    batch_name = f"{run_name}_{dataset}"
    loader = BatchChatGPTLoader(batch_name)
    loader.prepare_response()

    def predict(text):
        prompt = p_builder.get_prompt(text)
        ret_text = loader.get_response(prompt)
        pred = p_builder.get_label_from_response(ret_text)
        print(ret_text)
        return pred, 0

    return predict


def predict_clf_main(
        run_name: str,
        dataset: str,
) -> None:
    predict_fn = get_offline_clf_predictor(run_name, dataset)
    clf_predict_w_predict_fn(dataset, run_name, predict_fn)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
