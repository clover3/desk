import logging
import random
from abc import abstractmethod, ABC
from time import time
from typing import Callable
from typing import List, Tuple, Any
from sklearn.metrics import auc, roc_curve

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from chair.list_lib import left, right
from toxicity.llama_guard.make_edit_payload import convert_to_binary
from toxicity.llama_guard.output_convertor import parse_prediction_paired

LOG = logging.getLogger(__name__)


def generate(model, tok, prompt: str, device) -> np.array:
    seed_everything(42)
    prompt_tok = tok(
        prompt,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    output = model.generate(
        input_ids=prompt_tok['input_ids'],
        attention_mask=prompt_tok['attention_mask'],
        max_new_tokens=20,
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    _, start_index = prompt_tok["input_ids"].shape

    first_token_score = output.scores[0]
    generated_seq = output.sequences[0][start_index:]
    unsafe_score = first_token_score[0][generated_seq[0]].cpu().numpy()

    def decode(seq):
        raw_str = tok.decode(seq, skip_special_tokens=True)
        return raw_str

    pred_s = decode(generated_seq)
    return pred_s, unsafe_score


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_model(hparams):
    assert hparams is not None, 'Error: hparams is None.'
    model_name = hparams.model_name
    LOG.info("Instantiating model")
    if isinstance(model_name, str):
        device_map = 'auto' if hparams.model_parallel else None
        torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32

        if 't5' in model_name.lower():
            model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype,
                                                               device_map=device_map)
            tok = T5Tokenizer.from_pretrained(model_name)
        elif 'gpt-3.5' in model_name.lower():
            model, tok = None, None
        elif 'gpt' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
            tok = GPT2Tokenizer.from_pretrained(model_name)
            tok.pad_token_id = tok.eos_token_id
        elif 'llama' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
            tok = AutoTokenizer.from_pretrained(model_name)
            tok.pad_token_id = tok.eos_token_id
        elif 'baichuan' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True,
                                                         device_map=device_map)
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tok.pad_token_id = tok.eos_token_id
        elif 'chatglm' in model_name.lower():
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                              device_map=device_map)
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tok.unk_token_id = 64787
        elif 'internlm' in model_name.lower():
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                              device_map=device_map)
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tok.pad_token_id = tok.eos_token_id
        elif 'qwen2' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                         torch_dtype=torch_dtype if hparams.alg_name not in [
                                                             'MEND'] else torch.bfloat16, device_map=device_map)
            tok = AutoTokenizer.from_pretrained(model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',
                                                unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'qwen' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, fp32=False, trust_remote_code=True,
                                                         device_map=device_map)
            tok = AutoTokenizer.from_pretrained(model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',
                                                unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'mistral' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
            tok = AutoTokenizer.from_pretrained(model_name)
            tok.pad_token_id = tok.eos_token_id
        else:
            raise NotImplementedError

        if tok is not None and (isinstance(tok, GPT2Tokenizer) or isinstance(tok, GPT2TokenizerFast) or isinstance(tok,
                                                                                                                   LlamaTokenizer)) and (
                hparams.alg_name not in ['ROME', 'MEMIT']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
            tok.padding_side = 'left'
        if tok is not None and (
                'mistral' in model_name.lower() or 'llama' in model_name.lower() or 'qwen' in model_name.lower()) and (
                hparams.alg_name in ['ROME', 'MEMIT']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
            tok.padding_side = 'right'
    else:
        model, tok = model_name

    if hparams.model_parallel:
        hparams.device = str(model.device).split(":")[1]
    if not hparams.model_parallel and hasattr(hparams, 'device'):
        model.to(f'cuda:{hparams.device}')
    return model, tok


class EditorSpec(ABC):
    @abstractmethod
    def batch_edit(self, model, tokenizer, edit_payload):
        pass


Model = Any
Tokenizer = Any
EditSpec = Callable[[Model, Tokenizer, List[tuple[str, str]]], Tuple[Model, Tokenizer]]


def text_compare(pred_s, target_new):
    is_correct = (pred_s.strip() == target_new.strip())

    def escape(s):
        return s.encode('unicode_escape').decode()

    LOG.debug(f"labels/pred/is_correct={escape(target_new)}/{escape(pred_s)}/{is_correct}")
    result = np.array([is_correct], np.float32)
    return result



class EditExperimentRunner:
    def __init__(self, hparams, edit_fn: EditSpec):
        self.model, self.tok = initialize_model(hparams)
        self.device = str(self.model.device).split(":")[1]
        self.edit_fn: EditSpec = edit_fn

    def edit_and_eval(self,
                      edit_payload: List[tuple[str, str]],
                      valid_data: List[tuple[str, str]] = None,
                      eval_data_d=None,
                      skip_eval=False,
                      ):
        do_pre_eval = not skip_eval
        do_post_eval = not skip_eval
        return self.run_edit_and_eval2(edit_payload, valid_data, eval_data_d, do_pre_eval, do_post_eval)

    def run_edit_and_eval2(self,
                 edit_payload: List[tuple[str, str]],
                 valid_data: List[tuple[str, str]] = None,
                 eval_data_d=None,
                 do_pre_eval=False,
                 do_post_eval=False):
        if eval_data_d is None:
            eval_data_d = {}
        seed_everything(42)
        eval_todo = {
            "train": edit_payload,
        }
        if valid_data is not None:
            eval_todo["valid"] = valid_data
        eval_todo.update(eval_data_d)

        def eval_datasets(model):
            metric_d = {}
            for eval_split, eval_data in eval_todo.items():
                start = time()
                prompt_list: list[str] = left(eval_data)
                label_str: list[str] = right(eval_data)
                pred_tuples: list[tuple[str, float]] = \
                    [generate(model, self.tok, p, self.device) for p in prompt_list]
                labels_int: list[int] = [convert_to_binary(e, "unsafe") for e in label_str]
                acc = np.mean([text_compare(p, l) for p, l in zip(left(pred_tuples), label_str)])
                parsed: list[tuple[str, float]] = parse_prediction_paired(pred_tuples, "unsafe", )
                scores = right(parsed)
                fpr, tpr, thresholds = roc_curve(labels_int, scores)
                auc_val = auc(fpr, tpr)
                metric_d[eval_split + "_acc"] = acc
                metric_d[eval_split + "_auc"] = auc_val
                LOG.info(f"{eval_split} acc\t{acc:.3f}")
                LOG.info(f"{eval_split} auc\t{auc_val:.3f}")
                LOG.info(f"Evaluation took {time() - start:.2f} sec")
            return metric_d

        LOG.info(f"Pre-edit eval")
        pre_metric_d = eval_datasets(self.model) if do_pre_eval else {}
        start = time()
        edited_model = self.edit_fn(self.model, self.tok, edit_payload)
        exec_time = time() - start
        LOG.info(f"Editing took {exec_time:.1f}")
        LOG.info(f"Post-edit eval")
        post_metric_d = eval_datasets(self.model) if do_post_eval else {}
        metrics = {
            "pre": pre_metric_d,
            "post": post_metric_d
        }
        return metrics, edited_model
