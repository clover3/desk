import os

from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_user.llama_helper.lf_local import LlamaClient
from llama_user.llama_helper.llama_model_names import Llama3_8B_Instruct
from rule_gen.reddit.llama.prompt_helper import get_prompt_fn_from_type


def load_llama_inst(model_id: str = Llama3_8B_Instruct):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    def request(prompt: str) -> tuple[str, float]:
        chat = tokenizer.apply_chat_template([{"role": "system", "content": prompt}], tokenize=False)
        input = tokenizer([chat], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=20, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        score = output.scores[0]
        score = score[0][generated_seq[0]]
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        return result, score.cpu().numpy()

    return request


def get_lf_classifier(model_id, get_prompt, sb):
    pos_keyword = "yes"
    client = LlamaClient(model_id, max_prompt_len=5000)

    def predict(text):
        prompt = get_prompt(text, sb)
        ret_text = client.ask("", prompt)
        pred = int(pos_keyword.lower() in ret_text.lower())
        return pred, 0

    return predict


def get_lf_predictor_w_conf(run_name):
    model_name, sb = run_name.split("/")
    conf_path = os.path.join("confs", "lf", f"{model_name}.yaml")
    conf = OmegaConf.load(conf_path)
    get_prompt_fn = get_prompt_fn_from_type(conf.prompt_type)
    return get_lf_classifier(conf.model_path, get_prompt_fn, sb)


