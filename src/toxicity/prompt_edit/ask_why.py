from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.llama_helper.lf_client import LLMClient
from toxicity.misc_lib import SuccessCounter


# Iterate Prompt of Toxigen
# If wrong, put that into system prompt

def main():
    split = "train"
    test_dataset: ToxigenBinary = ToxigenBinary(split)
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += " If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    key_string = "toxic"
    client = LLMClient()

    for idx, e in enumerate(test_dataset):
        if idx < 10:
            continue
        text = e["text"]
        ret = client.ask(text, instruction)
        label = int(key_string in ret)
        label_str = ["benign", "toxic"][e["label"]]
        is_correct = e["label"] == label
        print("===")
        print(f"Data {e['id']}. is_correct={is_correct} ({label_str})")

        if not is_correct:
            print("text: ", text)
            prompt = (f"This text is considered {label_str}. Could you guess why? Then make a rule e.g, A text should be toxic if ... \n"
                      f"Text: {text}")
            ret = client.ask(prompt)
            print(ret)


if __name__ == "__main__":
    main()