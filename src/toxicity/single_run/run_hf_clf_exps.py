from toxicity.hf_classifiers.get_clf import toxicity_hf_model_names
from desk_util.runnable import clf_predict_inner
from desk_util.runnable.run_eval_clf import run_eval_clf


def main():
    todo = ["toxigen_head_100_para_clean", "toxigen_test_head_100"]
    todo = ["toxigen_train_head_100"]
    for dataset in todo:
        for run_name in toxicity_hf_model_names.keys():
            clf_predict_inner(dataset, run_name)
            run_eval_clf(run_name, dataset, True)


if __name__ == "__main__":
    main()
