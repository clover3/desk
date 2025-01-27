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

