from datasets import load_dataset

from llama_user.llama_helper.lf_client import LLMClient


# Iterate Prompt of Toxigen
# If wrong, put that into system prompt

def main():
    all_data = load_dataset("toxigen/toxigen-data", name="annotated")["train"]
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += " \n"
    instruction += ("\n Compare these sentences how much they are toxic. "
                    "Assign toxicity scores for each of them from 1 (not toxic) to 5 (very toxic)")

    client = LLMClient()

    base_idx = 0

    batch_size = 4
    for base_idx in range(0, 12, 2):
        sents = []
        for i in range(batch_size):
            e = all_data[base_idx+i]
            text = e["text"]
            sents.append(f"{i+1}: {text}")
        text_formed = "\n".join(sents)
        print(text_formed)
        ret = client.ask(text_formed, instruction)
        print("===")
        print(ret)


if __name__ == "__main__":
    main()