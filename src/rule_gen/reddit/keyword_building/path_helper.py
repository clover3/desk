import json
import os

from rule_gen.cpath import output_root_path


def get_keyword_statement_path(name):
    parsed_path = os.path.join(
        output_root_path, "reddit", "rule_processing", "keyword_statement", f"{name}.json")
    return parsed_path


def get_keyword_req_response_path(sb):
    raw_path = os.path.join(
        output_root_path, "reddit", "rule_processing", "keyword_raw", f"{sb}.json")
    return raw_path


def get_parsed_keyword_path(sb):
    parsed_path = os.path.join(
        output_root_path, "reddit", "rule_processing", "keyword", f"{sb}.json")
    return parsed_path


def get_named_keyword_path(name, sb):
    save_path = os.path.join(
        output_root_path, "reddit", "rule_processing", f"keyword_{name}", f"{sb}.json")
    return save_path


def get_named_keyword_statement_path(name, sb):
    parsed_path = os.path.join(
        output_root_path, "reddit", "rule_processing", f"keyword_statement_{name}", f"{sb}.json")
    return parsed_path


def load_keyword_statement(sb) -> list[tuple[str, str]]:
    parsed_path = get_keyword_statement_path(sb)
    return json.load(open(parsed_path, "r"))



def load_named_keyword_statement(name, sb) -> list[tuple[str, str]]:
    parsed_path = get_named_keyword_statement_path(name, sb)
    return json.load(open(parsed_path, "r"))


def get_entail_save_path(name, sb):
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", f"k_{name}_to_text_100", f"{sb}.csv")
    return res_save_path


def get_statements_from_ngram(sb):
    statement_path = os.path.join(
        output_root_path, "reddit",
        "ngram_based", f"{sb}.json")
    return statement_path
