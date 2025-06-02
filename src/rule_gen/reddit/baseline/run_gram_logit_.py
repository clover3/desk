import os

import fire

from rule_gen.cpath import output_root_path
from rule_gen.reddit.baseline.load_data_tokenized import load_reddit_train_data_tokenized
from rule_gen.reddit.baseline.ngram_logit import NgramLogisticRegression


def main(sb="TwoXChromosomes"):
    data_name = "train_data2"
    save_dir_name = "ngram_logit"
    print("Running ngram logit")
    print("Tokenizing")
    run_ngramlogit_train(data_name, save_dir_name, sb)


def run_ngramlogit_train(data_name, save_dir_name, sb):
    train_data = load_reddit_train_data_tokenized(data_name, sb, "train")
    val_data = load_reddit_train_data_tokenized(data_name, sb, "val")
    # Test different2 n-gram sizes
    n_list = [1, 2, 3]
    # Create and train model
    model = NgramLogisticRegression(n=n_list)
    model.fit(train_data)
    model_path = os.path.join(output_root_path, "models", save_dir_name, f"{sb}.pickle")
    model.save(model_path)
    model = NgramLogisticRegression.load_model(model_path)
    # Evaluate model
    results = model.evaluate(val_data)
    print_metrics = ["accuracy", "f1"]
    for metric in print_metrics:
        print(f"{metric}\t{results[metric]}")
    top_features = model.get_feature_importance(top_n=5)
    print("Top features:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
