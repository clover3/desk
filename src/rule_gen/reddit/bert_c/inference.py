import torch
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_c.c2_modeling import BertC2
from rule_gen.reddit.bert_c.c_modeling import BertC1
from rule_gen.reddit.bert_c.train_w_sb_ids import load_sb_name_to_id_mapping

global_d = {}
def get_bert_c_predictor_by_run_name(run_name):
    model_name, sb_name = run_name.split("/")
    if model_name.startswith("bert_c1"):
        return get_bert_c_predictor(BertC1, run_name)

    elif model_name.startswith("bert_c2"):
        return get_bert_c_predictor(BertC2, run_name)


def get_bert_c_predictor(model_cls, run_name):
    global global_d
    model_name, sb_name = run_name.split("/")
    max_length = 256
    model_path = get_model_save_path(model_name)
    if model_name not in global_d:
        model = model_cls.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # Move model to appropriate device
        global_d[model_path] = model
    else:
        model = global_d[model_name]
    sb_name_dict = load_sb_name_to_id_mapping()
    sb_id = sb_name_dict[sb_name]

    def predict(text):
        inputs = model.tokenizer(
            text, padding='max_length',
            truncation=True, max_length=max_length, return_tensors="pt")
        inputs["sb_id"] = torch.tensor([sb_id], dtype=torch.long)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            c2_output = model(**inputs, return_dict=True)

        score = c2_output.logits.cpu().numpy()
        score = score[0]
        label = int(score > 0)
        return label, score

    return predict
