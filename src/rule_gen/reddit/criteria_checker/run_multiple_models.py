from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    subreddit_list = get_split_subreddit_list("train")
    n_sb = 60

    predict_fn_d = {}
    while True:
        try:
            comment = input("Enter comment: ")
            outcome = {1: [], 0: []}
            for i, sb in enumerate(subreddit_list[:n_sb]):
                run_name = "bert2_{}".format(sb)
                if run_name not in predict_fn_d:
                    print("Loading {}".format(run_name))
                    predict_fn = get_classifier(run_name)
                    predict_fn_d[run_name] = predict_fn
                else:
                    predict_fn = predict_fn_d[run_name]

                pred, _score = predict_fn(comment)
                pred = int(pred)

                outcome[pred].append(sb)
            print(outcome)
        except Exception as e:
            print(e)




if __name__ == "__main__":
    main()

