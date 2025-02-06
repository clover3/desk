import logging
import time
from typing import Any

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModel, AutoModelForCausalLM
import math

from transformers.modeling_outputs import BaseModelOutputWithPast

from chair.list_lib import list_equal
from desk_util.io_helper import init_logging

LOG = logging.getLogger(__name__)


class TrieNode:
    def __init__(self, parent_ids=[]):
        self.children = {}  # token -> TrieNode
        self.children_key_tensor = None
        self.is_end = False
        self.parent_ids: list = parent_ids

    def set_children_key_tensor(self, device):
        keys = list(self.children.keys())
        self.children_key_tensor = torch.tensor(keys, device=device)

def build_trie(post_fix_token_ids):
    root = TrieNode()

    # Build trie from each post_fix sequence
    for token_seq in post_fix_token_ids:
        node = root
        parent_ids = []
        for token in token_seq:
            parent_ids.append(token)
            if token not in node.children:
                node.children[token] = TrieNode(list(parent_ids))

            node = node.children[token]
        node.is_end = True

    # For debugging
    LOG.debug("Built trie structure:")

    def print_trie(node, depth=0, path=[]):
        if node.is_end:
            LOG.debug(f"Complete sequence: {path}")
        for token, child in node.children.items():
            LOG.debug(f"{'  ' * depth}Token {token} -> has {len(child.children)} children")
            print_trie(child, depth + 1, path + [token])

    print_trie(root)
    return root


class LlamaProbabilityCalculator:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_log_probability(self, prefix: str, post_fix_list: list[str]) -> list[float]:
        """
        Compute log probabilities for each postfix string given the prefix.

        Args:
            prefix (str): The context string to condition on
            post_fix_list (list[str]): List of possible continuations to compute log probabilities for

        Returns:
            list[float]: List of log probabilities corresponding to each postfix string
        """
        # Encode the prefix
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Get the logits for the last token
        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]

        # Convert logits to log probabilities using log_softmax
        next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        log_probabilities = []

        for postfix in post_fix_list:
            # Tokenize the postfix
            postfix_tokens = self.tokenizer.encode(postfix, add_special_tokens=False)

            if not postfix_tokens:
                log_probabilities.append(float('-inf'))
                continue

            # Get log probability of the first token of the postfix
            first_token_log_prob = next_token_log_probs[0, postfix_tokens[0]].item()

            # If postfix is multiple tokens, compute conditional log probability
            if len(postfix_tokens) > 1:
                # Create input sequence by concatenating prefix and postfix
                full_sequence = prefix + postfix
                full_inputs = self.tokenizer(full_sequence, return_tensors="pt").to(self.device)
                full_ids = full_inputs["input_ids"]

                # Get the starting position of postfix tokens
                prefix_length = len(input_ids[0])

                # Generate outputs for the full sequence
                with torch.no_grad():
                    full_outputs = self.model(full_ids)
                    full_logits = full_outputs.logits

                # Compute conditional log probabilities for remaining tokens
                sequence_log_prob = first_token_log_prob
                for i in range(1, len(postfix_tokens)):
                    position = prefix_length + i - 1
                    next_token_logits = full_logits[:, position, :]
                    next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                    token_log_prob = next_token_log_probs[0, postfix_tokens[i]].item()
                    sequence_log_prob += token_log_prob
                log_probabilities.append(sequence_log_prob)
            else:
                # Single token case
                log_probabilities.append(first_token_log_prob)

        return log_probabilities

    def compute_pair_probability_raw(self, prefix: str, post_fix_list: list[str]):
        st = time.time()
        probs = self.compute_log_probability(prefix, post_fix_list)
        ed = time.time()
        LOG.info("compute_probability took {}".format(ed-st))
        diff_sorted = []
        for i in range(0, len(post_fix_list), 2):
            p1 = probs[i]
            p2 = probs[i+1]
            diff_sorted.append((i // 2, post_fix_list[i], p1 - p2))
            LOG.info(f"Result ({i}): '{post_fix_list[i]}' (log prob: {p1:.4f})")
            LOG.info(f"Result ({i+1}): '{post_fix_list[i+1]}' (log prob: {p2:.4f})")

        diff_sorted.sort(key=lambda x: x[2], reverse=True)
        LOG.info("\nDiff sorted results:")
        for idx, text, diff in diff_sorted:
            LOG.info(f"Result ({idx}): '{text}' (log prob diff: {diff:.4f})")

        return diff_sorted


    def compute_probability_trie(self, prefix: str, post_fix_list: list[str]):
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        # Tokenize the post_fix_list
        post_fix_token_ids = [self.tokenizer.encode(post_fix, add_special_tokens=False) for post_fix in post_fix_list]
        trie = build_trie(post_fix_token_ids)

        def traverse_and_set_tensors(node, device):
            if node.children:
                node.set_children_key_tensor(device)
                for child in node.children.values():
                    traverse_and_set_tensors(child, device)

        traverse_and_set_tensors(trie, self.device)

        # Initialize beam with just the prefix sequence
        beams: list[tuple[Any, Any, TrieNode]] = [(input_ids, 0.0, trie)]  # (sequence, log_prob, node in trie)
        LOG.info(f"Starting probability computation with prefix: {prefix}")
        LOG.info(f"Initial input_ids shape: {input_ids.shape}")
        beam_width = 100
        results = []
        st_all = time.time()
        ed = None
        while beams:
            new_beams: list[tuple] = []
            LOG.debug(f"Current beam has: {len(beams)}")
            for beam_ids, beam_log_prob, node in beams:
                with torch.no_grad():
                    # st = time.time()
                    # if ed is not None:
                    #     LOG.info(f"Other routine took: {st-ed}")
                    outputs = self.model(beam_ids)
                    # ed = time.time()
                    # LOG.info(f"Model call took: {ed-st}")
                    next_token_logits = outputs.logits[0, -1]
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    st1 = time.time()
                    # valid_tokens = node.children_key_tensor
                    valid_tokens = list(node.children.keys())
                    valid_log_probs = next_token_log_probs[valid_tokens]
                    ed1 = time.time()
                    # LOG.info(f"next_token_log_probs[valid_tokens]: {ed1-st1}")
                    valid_tokens_s = [self.tokenizer.decode(v) for v in valid_tokens]

                    # LOG.debug(f"Valid tokens after: {self.tokenizer.decode(node.parent_ids)}")
                    # LOG.debug(f"{valid_tokens_s}")

                    for token, log_prob in zip(valid_tokens, valid_log_probs):
                        token_tensor = torch.tensor([[token]], device=self.device)
                        new_ids = torch.cat([beam_ids, token_tensor], dim=1)
                        new_log_prob = beam_log_prob + log_prob.item()

                        # Check if this matches any complete post_fix sequence
                        next_node = node.children[int(token)]
                        if next_node.is_end:
                            post_fix_ids = list(next_node.parent_ids)
                            results.append((new_ids, new_log_prob, post_fix_ids))
                        else:
                            new_beams.append((new_ids, new_log_prob, next_node))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        elapsed = time.time() - st_all
        LOG.info(f"Completed probability computation. elapsed={elapsed}")
        # Sort results by log probability (higher is better)
        # results.sort(key=lambda x: x[1], reverse=True)

        output = []
        for idx, (ids, log_prob, post_fix_ids) in enumerate(results):
            match_idx = None
            postfix = ids[0].tolist()[len(input_ids[0]):]
            for c_idx, cand in enumerate(post_fix_token_ids):
                if list_equal(cand, postfix):
                    match_idx = c_idx
            text = self.tokenizer.decode(post_fix_ids)
            output.append((match_idx, text, log_prob))
        output.sort()
        LOG.info("\nPaired results:")
        diff_sorted = []
        for idx in range(0, len(output), 2):
            src_idx1, text1, new_log_prob_1 = output[idx]
            src_idx2, text2, new_log_prob_2 = output[idx + 1]
            LOG.info(f"Result ({src_idx1}): '{text1}' (log prob: {new_log_prob_1:.4f})")
            LOG.info(f"Result ({src_idx2}): '{text2}' (log prob: {new_log_prob_2:.4f})")
            diff_sorted.append((src_idx1 // 2, text1, new_log_prob_1 - new_log_prob_2))

        diff_sorted.sort(key=lambda x: x[2], reverse=True)
        LOG.info("\nDiff sorted results:")
        for idx, text, diff in diff_sorted:
            LOG.info(f"Result ({idx}): '{text}' (log prob diff: {diff:.4f})")

        return diff_sorted

    def beam_search(self, prefix: str, beam_width: int = 3, max_tokens: int = 20):
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Initialize beam with just the prefix sequence
        beams = [(input_ids, 0.0)]  # (sequence, log_prob)
        LOG.info(f"Starting beam search with prefix: {prefix}")
        LOG.info(f"Initial input_ids shape: {input_ids.shape}")

        for step in range(max_tokens):
            LOG.info(f"\nStep {step + 1}/{max_tokens}")
            candidates = []

            # Expand each beam
            for beam_idx, (beam_ids, beam_log_prob) in enumerate(beams):
                LOG.info(f"Processing beam {beam_idx + 1}/{len(beams)}, current log prob: {beam_log_prob:.4f}")

                with torch.no_grad():
                    outputs = self.model(beam_ids)
                    next_token_logits = outputs.logits[0, -1]
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    top_log_probs, top_tokens = torch.topk(next_token_log_probs, beam_width)

                    LOG.info(f"Top {beam_width} tokens for beam {beam_idx + 1}:")
                    for token_idx, (token, log_prob) in enumerate(zip(top_tokens, top_log_probs)):
                        token_text = self.tokenizer.decode([token.item()])
                        LOG.info(
                            f"  Token {token_idx + 1}: '{token_text}' (id: {token.item()}, log prob: {log_prob:.4f})")

                        new_ids = torch.cat([beam_ids, token.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_log_prob = beam_log_prob + log_prob.item()
                        candidates.append((new_ids, new_log_prob))

            # Select top beams for next iteration
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            LOG.info("\nSelected top beams for next iteration:")
            for beam_idx, (beam_ids, beam_log_prob) in enumerate(beams):
                beam_text = self.tokenizer.decode(beam_ids[0], skip_special_tokens=True)
                LOG.info(f"Beam {beam_idx + 1}: '{beam_text}' (log prob: {beam_log_prob:.4f})")

            # Stop if all beams end with EOS
            if all(self.tokenizer.eos_token_id in beam[0][0] for beam in beams):
                LOG.info("\nAll beams ended with EOS token, stopping early")
                break

        # Convert to text and return with scores
        results = []
        for beam_ids, beam_log_prob in beams:
            text = self.tokenizer.decode(beam_ids[0], skip_special_tokens=True)
            results.append((text, beam_log_prob))

        LOG.info("\nFinal results:")
        for idx, (text, log_prob) in enumerate(results):
            LOG.info(f"Result {idx + 1}: '{text}' (log prob: {log_prob:.4f})")
        return results


def main1():
    calculator = LlamaProbabilityCalculator()
    comment = "Most scientologists believe psychiatry is a scam"
    template = "This text is contains "
    prefix = comment + ".\n " + template
    postfixes = [" hostility", " subreddit", " a statement of fact"]
    results = calculator.compute_probability_trie(prefix, postfixes)
    print("\nRanked sequences by log probability:")
    for sequence, log_prob in results:
        print(f"{sequence}: {log_prob:.4f}")


def main2():
    init_logging()
    calculator = LlamaProbabilityCalculator()
    comment = "Most scientologists believe psychiatry is a scam"
    template = "This text contains "
    prefix = comment + ".\n " + template
    results = calculator.beam_search(prefix, beam_width=3, max_tokens=10)
    for text, score in results:
        print(f"{text}: {score:.4f}")


if __name__ == "__main__":
    main2()

