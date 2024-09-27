from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


model_name = "s-nlp/roberta_toxicity_classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

for name, param in model.named_parameters():
    print(name, param.shape)
