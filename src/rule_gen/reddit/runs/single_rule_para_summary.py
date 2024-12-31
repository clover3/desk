from rule_gen.reddit.runs.single_rule_summary import single_run_result


def single_rule_result(rule_idx, rule_sb):
    run_name = f"api_srr_{rule_sb}_{rule_idx}_detail"
    single_run_result(run_name)

def main():
    rule_sb = "TwoXChromosomes"
    rule_idx = 0
    single_rule_result(rule_idx, rule_sb)


if __name__ == "__main__":
    main()
