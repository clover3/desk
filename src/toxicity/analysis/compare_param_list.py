import gc
from transformers import AutoModelForCausalLM

import torch

from toxicity.llama_helper.llama_model_names import failspy, Llama3_8B_Instruct

gc.collect()
torch.cuda.empty_cache()

def get_named_param_list(model_name):
    # Assuming you've already loaded your model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Enumerate and print the names of the variables (parameters)
    output = []
    for name, param in model.state_dict():
        output.append((name, param.shape))
    return output


def get_param_list(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return list(model.parameters())


def main():
    l1 = get_named_param_list(Llama3_8B_Instruct)
    l2 = get_named_param_list(failspy)
    print(len(l1))
    print(l2)

    d1 = dict(l1)
    d2 = dict(l2)
    new_param = [k for k in d2.keys() if k not in d1]
    print(new_param)


def main():
    l1 = get_param_list(Llama3_8B_Instruct)
    l2 = get_param_list(failspy)
    print(len(l1))
    print(len(l2))



if __name__ == "__main__":
    main()
