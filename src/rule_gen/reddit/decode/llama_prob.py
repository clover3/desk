import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import math


class LlamaProbabilityCalculator:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_logits_for_sequence(self, sequence, temperature=1.0):
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
            return outputs.logits / temperature, inputs["input_ids"]

    def compute_sequence_log_probability(self, sequence, temperature=1.0, cached_prefix_output=None):
        try:
            if cached_prefix_output is None:
                logits, input_ids = self.get_logits_for_sequence(sequence, temperature)
            else:
                prefix_len, logits = cached_prefix_output
                inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]
                if prefix_len > 0:
                    with torch.no_grad():
                        new_outputs = self.model(input_ids[:, prefix_len:])
                        logits = torch.cat([logits[:, :prefix_len], new_outputs.logits], dim=1)

            log_probs = torch.log_softmax(logits, dim=-1)
            total_log_prob = 0.0
            for i in range(input_ids.shape[1] - 1):
                next_token_id = input_ids[0, i + 1]
                total_log_prob += log_probs[0, i, next_token_id].item()

            return total_log_prob, (input_ids.shape[1], logits)

        except Exception as e:
            print(f"Error computing log probability: {str(e)}")
            return None, None

    def compute_probability(self, prefix: str, post_fix_list: list[str]):
        results = []

        # Compute prefix logits once
        prefix_log_prob, cached_output = self.compute_sequence_log_probability(prefix)

        for postfix in post_fix_list:
            full_sequence = f"{prefix}{postfix}"
            log_prob, _ = self.compute_sequence_log_probability(full_sequence, cached_prefix_output=cached_output)
            results.append((full_sequence, log_prob))

        results.sort(key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
        return results


    def beam_search(self, prefix: str, beam_width: int = 3, max_tokens: int = 20):
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Initialize beam with just the prefix sequence
        beams = [(input_ids, 0.0)]  # (sequence, log_prob)

        for _ in range(max_tokens):
            candidates = []

            # Expand each beam
            for beam_ids, beam_log_prob in beams:
                with torch.no_grad():
                    outputs = self.model(beam_ids)
                    next_token_logits = outputs.logits[0, -1]
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                    # Get top-k next tokens
                    top_log_probs, top_tokens = torch.topk(next_token_log_probs, beam_width)

                    for token, log_prob in zip(top_tokens, top_log_probs):
                        new_ids = torch.cat([beam_ids, token.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_log_prob = beam_log_prob + log_prob.item()
                        candidates.append((new_ids, new_log_prob))

            # Select top beams for next iteration
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop if all beams end with EOS
            if all(self.tokenizer.eos_token_id in beam[0][0] for beam in beams):
                break

        # Convert to text and return with scores
        results = []
        for beam_ids, beam_log_prob in beams:
            text = self.tokenizer.decode(beam_ids[0], skip_special_tokens=True)
            results.append((text, beam_log_prob))

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
    calculator = LlamaProbabilityCalculator()
    comment = "Most scientologists believe psychiatry is a scam"
    template = "This text contains "
    prefix = comment + ".\n " + template
    results = calculator.beam_search(prefix, beam_width=3, max_tokens=10)
    for text, score in results:
        print(f"{text}: {score:.4f}")
    return NotImplemented


if __name__ == "__main__":
    main2()

