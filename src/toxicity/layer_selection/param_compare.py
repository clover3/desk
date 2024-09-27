import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from chair.list_lib import left
from chair.misc_lib import group_by

def parm_mag():
    model_id: str = "meta-llama/Meta-Llama-Guard-2-8B"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    def name_group(name):
        return ".".join(name.split(".")[3:5])

    name_to_v = {}
    for name, param in model.named_parameters():
        s = torch.norm(param).item()
        name_to_v[name] = s

    name_list = left(model.named_parameters())
    print(name_list)
    group = group_by(name_list, name_group)
    for g_name, items in group.items():
        for name in items:
            print(name, name_to_v[name])


def compare_models(model1, model2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compare two models by computing the difference vector for each parameter
    and summarizing these differences using multiple methods.
    Uses PyTorch for potentially faster computation, especially on GPU.
    """
    summaries = {}
    t = 0.01
    param2_d = dict(model2.named_parameters())
    for param_name, param1 in model1.named_parameters():
        param2 = param2_d[param_name]
        diff_vector = param1 - param2

        over_thres = torch.count_nonzero(t < torch.abs(diff_vector))
        # Summarize the difference vector
        summaries[param_name] = {
            'l1_norm': torch.norm(diff_vector, p=1).item(),
            'l2_norm': torch.norm(diff_vector, p=2).item(),
            'max_abs_diff': torch.max(torch.abs(diff_vector)).item(),
            "over_thres": over_thres.item(),
            'mean_diff': torch.mean(diff_vector).item(),
            'median_diff': torch.median(diff_vector).item(),
            'std_diff': torch.std(diff_vector).item()
        }

    return summaries

def main():

    model_id: str = "meta-llama/Meta-Llama-Guard-2-8B"
    model1 = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
    model_path = "/home/qdb5sn/work/LLaMA-Factory/saves/ft12"
    model2 = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

    summaries = compare_models(model1, model2)

    def name_group(name):
        return ".".join(name.split(".")[3:5])

    name_list = left(model1.named_parameters())
    print(name_list)
    group = group_by(name_list, name_group)

    metrics = None
    for param_name, metrics in summaries.items():
        metrics = metrics
        break

    for metric in metrics:
        print(metric)
        for g_name, items in group.items():
            for name in items:
                print(name, summaries[name])
        print()


if __name__ == "__main__":
    main()