from toxicity.llama_helper.lf_client import LLMClient


def main():
    client = LLMClient()
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += "If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    text = "If you were born in the US, you should be able to speak fluent English."
    prompt = f"{instruction}\n{text}"
    ret = client.ask(prompt)
    print(ret)


if __name__ == "__main__":
    main()

