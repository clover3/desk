from rule_gen.reddit.s9.token_scoring import get_predictor_from_run_name


def main():
    get_predictor_from_run_name("llama_s9_AskReddit")


if __name__ == "__main__":
    main()