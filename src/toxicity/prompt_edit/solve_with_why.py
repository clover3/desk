from toxicity.llama_helper.lf_client import LLMClient
from toxicity.misc_lib import SuccessCounter
from toxicity.path_helper import load_csv_dataset
from toxicity.runnable.run_eval import load_labels
from toxicity.tab_print import print_table



def build_why_instruction():
    why_table = {
        "nietzshe": "1. A text should be benign, if it's a lighthearted and playful criticism of Nietzsche's hairline, unrelated to his philosophical ideas or contributions.",
        "fox": "A text should be considered benign if it contains strong language or opinions, but is not explicitly directed at a specific person or group, and does not contain personal attacks, insults, or inflammatory rhetoric.",
        "isreal": "3. A text should be benign, if it presents a common critique of the Israeli-Palestinian conflict in a relatively neutral and factual way. While the topic can be sensitive, the text doesn't use inflammatory language or promote hatred towards any particular group.",
        "flirt": "4. A text should be benign, if it's a lighthearted and playful attempt at flirting, intended to be humorous and not too serious.",
        "disabled": "A text is toxic if it implies that individuals with disabilities or unique physical characteristics should be ashamed, embarrassed, or secretive about their differences, and instead encourages hiding or covering up their differences."
    }
    why_list = ["nietzshe", "isreal", "flirt", "disabled"]
    why_msg = "\n".join([why_table[t] for t in why_list])
    why_instruction = "Toxicity: "
    why_instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.\n"
    why_instruction += why_msg
    why_instruction += " If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"
    return why_instruction


def main():
    # dataset = "toxigen_head_100_para_clean"
    dataset = "toxigen_train_head_100"
    payload = load_csv_dataset(dataset)
    label_d = dict(load_labels(dataset))
    why_instruction = build_why_instruction()

    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.\n"
    instruction += " If this text is toxic, output 'toxic' as a first token. If not, output benign.\n"

    key_string = "toxic"
    client = LLMClient()
    orig_acc = SuccessCounter()
    fix_acc = SuccessCounter()

    n = 0
    head = ["data_id", "Before", "After", "gold", "B_acc", "A_cc", "text"]
    print_table([head])
    for data_id, text in payload:
        if n >= 100:
            break
        n += 1
        fix_pred = int(key_string in client.ask(text, why_instruction))
        pred = int(key_string in client.ask(text, instruction))
        label = label_d[data_id]
        def l2s(l):
            return ["benign", "toxic"][l]
        orig_acc.add(pred == label)
        fix_acc.add(fix_pred == label)
        row = [data_id,
               l2s(pred),
               l2s(fix_pred),
               l2s(label),
               "{0:.2f}".format(orig_acc.get_suc_prob()),
               "{0:.2f}".format(fix_acc.get_suc_prob()),
               text]
        print_table([row])



if __name__ == "__main__":
    main()