from typing import Callable

from rule_gen.reddit.s9.token_scoring import get_llama_criteria_scorer, get_s9_inst_target_seq_no_sb


def get_s9_combined() -> Callable[[str], list[int]]:
    scorer: Callable[[str], tuple[int, str, list[tuple[str, float]]]] = get_llama_criteria_scorer("llama_s9nosb")
    inst, target_seq = get_s9_inst_target_seq_no_sb()

    def text_to_binary(text: str) -> int:
        if text.lower() == "yes":
            return 1
        elif text.lower() == "no":
            return 0
        else:
            print("Unexpected text: ", text)
            return 0

    def get_feature(text) -> list[int]:
        pred, out_s, output = scorer(text)
        out_d: dict[str, int] = {code: text_to_binary(ans) for code, ans, score in output}
        print(out_d)
        try:
            binary_vector = []
            for key in target_seq:
                if key not in out_d:
                    v = 0
                else:
                    v = out_d[key]
                binary_vector.append(v)
        except KeyError:
            print("@@ Parse error:", out_s)
            binary_vector = [0] * len(target_seq)
        return binary_vector

    return get_feature
