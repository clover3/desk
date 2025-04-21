import math
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_user.llama_helper.lf_local import get_parsing_key



global_model_d = {}

class LlamaCriteriaScorer:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_prompt_len=10000, ):
        global global_model_d
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model_name not in global_model_d:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            global_model_d[model_name] = self.model
        else:
            self.model = global_model_d[model_name]
        self.max_prompt_len = max_prompt_len
        self.response_header = get_parsing_key(self.tokenizer)
        if not self.response_header:
            raise KeyError()
        self.truncate_count = 0

    def len_filter(self, text):
        if text is None:
            return text

        if len(text) < self.max_prompt_len:
            return text

        self.truncate_count += 1
        if self.truncate_count == 1 or math.log10(self.truncate_count).is_integer():
            print(f"Text has {len(text)} characters. Truncate to {self.max_prompt_len}")

        return text[:self.max_prompt_len]

    def ask(self, prompt, target_seq) -> tuple[str, list[tuple[str, float]]]:
        prompt = self.len_filter(prompt)
        j = self.get_json_request(prompt, None)
        n_retry = 2
        ret = None
        for _ in range(n_retry):
            try:
                ret = self._generate_response(j, target_seq)
                break
            except KeyError:
                pass
        if ret is None:
            ret = ("", [])
        return ret


    def _generate_response(self, request_json, target_seq) -> tuple[str, list[tuple[str, str, float]]]:
        input_text = self.tokenizer.apply_chat_template(
            request_json["messages"], tokenize=False,
            add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        output = self.model.generate(**inputs, max_new_tokens=100,
                                      return_dict_in_generate=True,
                                      output_scores=True,
                                      pad_token_id=pad_token_id)

        paired_scores = get_token_scores(self.model, self.tokenizer, output, prompt_len, target_seq)
        generated_seq = output.sequences[0][prompt_len:]
        gen_text = self.tokenizer.decode(generated_seq)
        return gen_text, paired_scores

    def get_json_request(self, prompt, system_prompt):
        if system_prompt is None:
            return {
                "model": self.model.config.name_or_path,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        else:
            return {
                "model": self.model.config.name_or_path,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }

def get_s9_inst_target_seq(sb):
    inst = f"The following text is a comment posted in {sb} subreddit."
    inst += f"Does this text violate any of the policy listed?\n"
    inst += f"Output in a format like 'S1 Yes S2 No ...', with no additional texts. \n"
    policy_list = """
S1 Toxic speech
S2 Hate speech
S3 Off topic
S4 Low-Quality
S5 Politics
S6 Moderation blame
S7 Advertising
S8 Sexual Content
S9 Link restriction
"""
    inst += policy_list
    inst += "\n Text: "
    target_seq = [f"S{i}" for i in range(1, 10)]
    return inst, target_seq


def get_predictor_from_run_name(run_name):
    tokens = run_name.split("_")
    p_name = tokens[1]
    sb = "_".join(tokens[2:])
    inst, target_seq = get_s9_inst_target_seq(sb)
    client = LlamaCriteriaScorer()
    def predict_fn(text):
        prompt = inst + text
        ret_text, scores = client.ask(prompt, target_seq)
        pred = "Yes" in ret_text
        return int(pred), ret_text, scores

    return predict_fn



def get_token_scores(model, tokenizer, output, prompt_len, target_seq):
    transition_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )
    token_scores = transition_scores[0].cpu().tolist()
    generated_tokens = output.sequences[0][prompt_len:]
    generated_token_ids = generated_tokens.cpu().tolist()
    token_texts = tokenizer.convert_ids_to_tokens(generated_token_ids)
    def clean_token(token):
        return token.replace('Ġ', '').replace('Ċ', '\n')
    clean_token_texts = [clean_token(token) for token in token_texts]
    t_idx = 0
    paired_scores = []
    parse_error = False
    for i, token in enumerate(clean_token_texts):
        if t_idx >= len(target_seq):
            break
        target_text = target_seq[t_idx]
        matched = True
        for j in range(len(target_text)):
            if i+j >= len(clean_token_texts) or not clean_token_texts[i+j] == target_text[j]:
                matched = False
        if matched:
            next_loc = i+len(target_text)
            if clean_token_texts[next_loc] in ["Yes", "No", "yes", "no"]:
                score = token_scores[next_loc]
                paired_scores.append((target_text, clean_token_texts[next_loc], score))
                t_idx += 1
            else:
                parse_error = True
                print("warning {} matched but next token is {}".format(target_text, clean_token_texts[next_loc]))

    if parse_error:
        token_score_pairs = list(zip(token_texts, transition_scores[0].tolist()))
        for token, score in token_score_pairs:
            print(f"Token: {token:15} Score: {score:.4f}")
        raise KeyError()
    return paired_scores
