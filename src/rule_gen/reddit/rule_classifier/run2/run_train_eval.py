import fire

from rule_gen.reddit.rule_classifier.run2.run_train_eval_on_all import SubredditClassifier


def main(sb):
    classifier = SubredditClassifier(
        sb,
    )
    train_score, val_score = classifier.train_and_evaluate()
    print(f"\nResults for subreddit: {sb}")
    print(f"Training F1 Score: {train_score:.4f}")
    print(f"Validation F1 Score: {val_score:.4f}")


if __name__ == "__main__":
    fire.Fire(main)

