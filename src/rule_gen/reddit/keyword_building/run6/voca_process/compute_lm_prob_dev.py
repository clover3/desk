import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_user.llama_helper.llama_model_names import Llama3_8B


def compute_text_log_probability_routine(text, model_name="gpt2"):
    """
    Compute the log probability of a given text using a causal language model.

    Args:
        text (str): The input text to compute log probability for
        model_name (str): The name of the model to use (default: "gpt2")

    Returns:
        float: The log probability of the text
    """
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return compute_text_log_prob(model, tokenizer, device, text)


def compute_text_log_prob(model, tokenizer, device, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    labels = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss.item()
    num_tokens = input_ids.size(1)
    log_prob = -loss * num_tokens
    return log_prob


# Example usage
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog."

    tokens = ('/', 'www', '.', 'sports', '-', 'stream', '.', 'net', '/', 'ch3', '.')
    text1 = " ".join(tokens)
    text2 = "".join(tokens)
    model_name = Llama3_8B  # You can replace with any causal LM from Hugging Face

    for t in (text1, text2):
        log_prob = compute_text_log_probability_routine(t, model_name)
        print(t)
        print(f"Log probability of text: {log_prob}")
    #
    # # To get per-token log probabilities
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokens = tokenizer.tokenize(sample_text)
    # print(f"Number of tokens: {len(tokens)}")
    # print(f"Average log probability per token: {log_prob / len(tokens)}")