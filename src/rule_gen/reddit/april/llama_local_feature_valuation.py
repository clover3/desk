import fire
import logging
from llama_user.llama_helper.lf_local import LlamaClient2
from rule_gen.reddit.april.feature_valuation import feature_valuation

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
    setup_log()
    max_text_len = 5000
    client = LlamaClient2(max_prompt_len=5000)
    logger = logging.getLogger("feature_valuation")

    pos_keyword = "Yes"

    def extract_feature(text):
        text = text[:max_text_len]
        prompt = f"{rule_text} Output as Yes/No."
        prompt += f"\n <text> {text} </text>"
        ret_text, score = client.ask(prompt)
        logger.debug(text)
        logger.debug("Res={} {}".format(ret_text, score))
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, score

    n_item = 300
    n_train = 200

    feature_valuation(extract_feature, n_item, n_train, sb)


if __name__ == "__main__":
    fire.Fire(main)
