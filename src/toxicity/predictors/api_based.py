from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.llama_helper.lf_client import LLMClient
from toxicity.prompt_edit.prompt_fix_exp import build_examples, load_predictions_parsed


def load_api_based(instruction):
    client = LLMClient()

    def check_conversation(conversation: list[str]) -> tuple[str, float]:
        text = conversation[-1]
        prompt = f"{instruction}\n{text}"
        ret = client.ask(prompt)
        return ret, 0
    return check_conversation


def load_api_based_system(instruction):
    client = LLMClient()

    def check_conversation(conversation: list[str]) -> tuple[str, float]:
        text = conversation[-1]
        ret = client.ask(text, instruction)
        return ret, 0
    return check_conversation


def load_api_based_system2(instruction_list):
    client = LLMClient()

    def check_conversation(conversation: list[str]) -> tuple[str, float]:
        text = conversation[-1]
        for instruction in instruction_list:
            text = client.ask(text, instruction)
        return text, 0
    return check_conversation

def load_api_1():
    instruction = get_api_1_prompt()
    return load_api_based(instruction)


def get_api_1_prompt():
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += "If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    return instruction


def load_api_2():
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += "Consider that the text is a part of a discussion possibly involving specific demographic groups.  "
    instruction += "If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    return load_api_based(instruction)


def load_api_3():
    n_item = 100
    preds = load_predictions_parsed("api_1", n_item)
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    test_dataset = list(test_dataset)[:n_item]

    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += " If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    instruction += "\n Here are examples \n"
    instruction += build_examples(test_dataset, preds)
    return load_api_based_system(instruction)


def load_api_4():
    why_msg = """
    1. A text should be benign, if it's a lighthearted and playful criticism of Nietzsche's hairline, unrelated to his philosophical ideas or contributions.
    2. A text should be benign, if it expresses a strong opinion about Fox News and Billboard, but does so in a way that's not aggressive, personal, or inflammatory.
    3. A text should be benign, if it presents a common critique of the Israeli-Palestinian conflict in a relatively neutral and factual way. While the topic can be sensitive, the text doesn't use inflammatory language or promote hatred towards any particular group.
    4. A text should be benign, if it's a lighthearted and playful attempt at flirting, intended to be humorous and not too serious.
    """
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += why_msg
    instruction += " If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    return load_api_based_system(instruction)


def load_api_5():
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += "Is this text include stereotypes over certain demographic groups? "
    instruction += "Considering these, generate explanation. The last word should be toxic or benign.\n"

    instruction2 = "If the text was considered toxic, output 'toxic' as a first token. If not, output benign"

    return load_api_based_system2([instruction, instruction2])


def load_api_based_predictor_by_name(run_name):
    tokens = run_name.split("_")
    base_name = "_".join(tokens[:2])
    method_d = {
        "api_1": load_api_1,
        "api_2": load_api_2,
        "api_3": load_api_3,
        "api_4": load_api_4,
        "api_5": load_api_5,
    }
    return method_d[base_name]()