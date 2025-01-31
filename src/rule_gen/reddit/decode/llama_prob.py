import logging
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
        self.is_end = False
        self.parent_ids: list = parent_ids

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
    LOG.info("Built trie structure:")

    def print_trie(node, depth=0, path=[]):
        if node.is_end:
            LOG.info(f"Complete sequence: {path}")
        for token, child in node.children.items():
            LOG.info(f"{'  ' * depth}Token {token} -> has {len(child.children)} children")
            print_trie(child, depth + 1, path + [token])

    print_trie(root)
    return root


class LlamaProbabilityCalculator:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_probability(self, prefix: str, post_fix_list: list[str]):
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Tokenize the post_fix_list
        post_fix_token_ids = [self.tokenizer.encode(post_fix, add_special_tokens=False) for post_fix in post_fix_list]
        trie = build_trie(post_fix_token_ids)
        # Initialize beam with just the prefix sequence
        beams: list[tuple[Any, Any, TrieNode]] = [(input_ids, 0.0, trie)]  # (sequence, log_prob, node in trie)
        LOG.info(f"Starting probability computation with prefix: {prefix}")
        LOG.info(f"Initial input_ids shape: {input_ids.shape}")
        beam_width = 100
        results = []

        while beams:
            new_beams = []
            LOG.debug(f"Current beam has: {len(beams)}")

            for beam_ids, beam_log_prob, node in beams:
                parent_str = self.tokenizer.decode(node.parent_ids)
                LOG.debug(f"beam_ids: {parent_str}")

                with torch.no_grad():
                    outputs = self.model(beam_ids)
                    next_token_logits = outputs.logits[0, -1]
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                    # Only consider tokens that are valid for the current position
                    valid_tokens = list(node.children.keys())
                    valid_log_probs = next_token_log_probs[valid_tokens]
                    valid_tokens_s = [self.tokenizer.decode(v) for v in valid_tokens]
                    LOG.debug(f"Valid tokens after: {parent_str}")
                    LOG.debug(f"{valid_tokens_s}")

                    for token, log_prob in zip(valid_tokens, valid_log_probs):
                        token_tensor = torch.tensor([[token]], device=self.device)
                        new_ids = torch.cat([beam_ids, token_tensor], dim=1)
                        new_log_prob = beam_log_prob + log_prob.item()

                        # Check if this matches any complete post_fix sequence
                        next_node = node.children[token]
                        if next_node.is_end:
                            post_fix_ids = list(next_node.parent_ids)
                            post_fix_ids.append(token)
                            results.append((new_ids, new_log_prob, post_fix_ids))
                        else:
                            new_beams.append((new_ids, new_log_prob, next_node))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # Sort results by log probability (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)

        output = []
        LOG.info("\nFinal results:")
        for idx, (ids, log_prob, post_fix_ids) in enumerate(results):
            match_idx = None
            postfix = ids[0].tolist()[len(input_ids[0]):]
            for c_idx, cand in enumerate(post_fix_token_ids):
                if list_equal(cand, postfix):
                    match_idx = c_idx
            text = self.tokenizer.decode(post_fix_ids)
            LOG.info(f"Result {idx + 1}: '{text}' (log prob: {log_prob:.4f})")
            output.append((match_idx, text, log_prob))

        return output

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
    results = calculator.compute_probability(prefix, postfixes)
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

