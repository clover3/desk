from llama_user.llama_helper.lf_client import LLMClient


def main():
    client = LLMClient()
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += "If this text is toxic, output 'toxic' as a first token. If not, output benign. Explain why\n"
    text = "Let's go Brendan."
    prompt = f"{instruction}\n{text}"
    ret = client.ask(prompt)
    print(ret)


if __name__ == "__main__":
    main()

