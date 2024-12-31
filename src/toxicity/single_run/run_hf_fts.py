from toxicity.hf_classifiers.ft import run_ft
from toxicity.hf_classifiers.get_clf import toxicity_hf_model_names, run_name_to_base_model_name
from desk_util.runnable import clf_predict_inner
from desk_util.runnable.run_eval_clf import run_eval_clf


def main():
    train_dataset = "toxigen_train_head_100"
    todo = toxicity_hf_model_names.keys()
    todo = ["christinacdl"]
    for base_model_prefix in todo:
        run_name = base_model_prefix + "_1"
        base_model_name = run_name_to_base_model_name(run_name)
        run_ft(train_dataset, base_model_name, run_name)
        for dataset in [train_dataset,
                        "toxigen_head_100_para_clean",
                        "toxigen_test_head_100"]:
            clf_predict_inner(dataset, run_name)
            run_eval_clf(run_name, dataset, True)


if __name__ == "__main__":
    main()
