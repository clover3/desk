import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import logging

from desk_util.path_helper import get_model_save_path

# Set logging level to ERROR to suppress warnings
logging.set_verbosity_error()
toxicity_hf_model_names = {
        "s-nlp": "s-nlp/roberta_toxicity_classifier",
        "MilaNLProc": "MilaNLProc/bert-base-uncased-ear-mlma",
        "cardiffnlp": "cardiffnlp/twitter-roberta-base-hate-latest",
        "Narrativaai": "Narrativaai/deberta-v3-small-finetuned-hate_speech18",
        "christinacdl": "christinacdl/olid_offensive_bert_multilingual"
    }

def classify_text(text, model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    result = classifier(text)
    return result


def get_result_to_int(model_name):
    key = {
        "s-nlp/roberta_toxicity_classifier": "toxic",
        "MilaNLProc/bert-base-uncased-ear-mlma": "hateful",
        "cardiffnlp/twitter-roberta-base-hate-latest": "HATE",
        "Narrativaai/deberta-v3-small-finetuned-hate_speech18": "HATE",
        "christinacdl/olid_offensive_bert_multilingual": "OFFENSIVE",
    }[model_name]

    def result_to_int(result):
        if key == result[0]["label"]:
            return 1
        else:
            return 0
    return result_to_int


def get_hf_classifier(run_name):

    try:
        model_name = toxicity_hf_model_names[run_name]
        base_model_name = model_name
    except KeyError:
        model_name = get_model_save_path(run_name)
        base_model_name = run_name_to_base_model_name(run_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    l2i = get_result_to_int(base_model_name)

    def classify(text):
        result = classifier(text)
        pred_label = l2i(result)
        score = result[0]["score"]
        return pred_label, score

    return classify


def run_name_to_base_model_name(run_name):
    base_model_name = toxicity_hf_model_names[run_name.split("_")[0]]
    return base_model_name


def main():
    models = [
        "s-nlp/roberta_toxicity_classifier",
        "MilaNLProc/bert-base-uncased-ear-mlma",
        "cardiffnlp/twitter-roberta-base-hate-latest",
        "Narrativaai/deberta-v3-small-finetuned-hate_speech18",
        "christinacdl/olid_offensive_bert_multilingual"
    ]

    text_to_classify = "We need to make slavery again for black people."
    print(text_to_classify)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model_name in models:
        print(f"\nClassifying with {model_name}:")
        try:
            result = classify_text(text_to_classify, model_name, device)
            print(result)
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()