from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


def process_masked_input(text, model_name="bert-base-uncased"):
    """
    Process text containing special tokens like [MASK] properly with BERT

    Args:
        text (str): Input text containing special tokens
        model_name (str): Name of the BERT model to use

    Returns:
        dict: Tokenized inputs ready for the model
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # The key is to use encode_plus() with add_special_tokens=True
    # and NOT manually add special tokens to the text
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    return encoded


# Example usage
def predict_masked_token(text, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Process input
    inputs = process_masked_input(text, model_name)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Find masked token position
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Get predicted token
    mask_token_logits = predictions[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1)

    # Convert to words
    for token_id in top_tokens.indices[0]:
        print(tokenizer.decode([token_id]))


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example usage:
text_with_mask = "The cat [unused97] on the mat."
inputs = process_masked_input(text_with_mask)
print(inputs["input_ids"][0])
print("Tokenized:", tokenizer.decode(inputs["input_ids"][0]))