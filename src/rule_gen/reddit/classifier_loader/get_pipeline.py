from transformers import pipeline
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.classifier_loader.torch_misc import get_device



def get_classifier_pipeline(run_name):
    model_path = get_model_save_path(run_name)
    pipe = pipeline("text-classification", model=model_path, device=get_device())

    label_map = {
        "LABEL_0": "0",
        "LABEL_1": "1",
    }

    def predict(text):
        try:
            r = pipe(text, truncation=True)[0]
            label = label_map[r["label"]]
            score = r["score"]
            if label == "0":
                score = -score
        except RuntimeError:
            label = "0"
            score = 0
        return label, score

    return predict


