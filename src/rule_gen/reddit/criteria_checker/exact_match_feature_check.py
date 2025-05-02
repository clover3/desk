import fire
from rule_gen.reddit.criteria_checker.feature_valuation import feature_valuation


def main(sb="askscience", feature: str =""):
    print(sb)

    def extract_feature(text):
        x_i = feature.lower() in text.lower()
        return x_i

    n_item = 3000
    n_train = 2000

    feature_valuation(extract_feature, n_item, n_train, sb)


if __name__ == "__main__":
    fire.Fire(main)
