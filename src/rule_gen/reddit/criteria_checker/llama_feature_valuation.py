import fire
import logging

from rule_gen.reddit.criteria_checker.feature_valuation import feature_valuation_over_train_data2


def setup_log():
    logger = logging.getLogger("feature_valuation")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.DEBUG)
    logging.info("Log set up")
    file_handler.flush()
    logger.addHandler(file_handler)



def main(sb="askscience", rule_text: str =""):
    print(sb)
    max_text_len = 5000
    from llama_user.llama_helper.lf_client import LLMClient
    client = LLMClient(max_prompt_len=5000)
    logger = logging.getLogger("feature_valuation")
    setup_log()

    pos_keyword = "Yes"

    def extract_feature(text):
        text = text[:max_text_len]
        prompt = f"{rule_text} Output as Yes/No."
        prompt += f"\n <text> {text} </text>"
        ret_text = client.ask(prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        logger.debug(text)
        logger.debug("Res={}".format(ret_text))

        ret = int(pred)
        return ret

    n_item = 300
    n_train = 200

    feature_valuation_over_train_data2(extract_feature, n_item, n_train, sb)


if __name__ == "__main__":
    fire.Fire(main)
