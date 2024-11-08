import fire

from toxicity.clf_util import load_csv_dataset_by_name
from toxicity.reddit.classifier_loader.prompt_based import PromptBuilder
from toxicity.apis.open_ai_batch_requester import BatchChatGPTSender, get_gpt_model_name_from_run_name


def predict_clf_main(
        run_name: str,
        dataset: str,
) -> None:
    batch_name = f"{run_name}_{dataset}"
    p_builder = PromptBuilder(run_name)

    model_name = get_gpt_model_name_from_run_name(run_name)
    requester = BatchChatGPTSender(batch_name, model_name)
    payload = load_csv_dataset_by_name(dataset)

    for e in payload:
        id, text = e
        request_id = f"{batch_name}_" + id
        prompt = p_builder.get_prompt(text)
        requester.add_request(request_id, prompt)

    requester.submit_request()


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
