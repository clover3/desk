import json
import os
from typing import Dict, Set

from rule_gen.reddit.path_helper import get_rp_path


def parse_data_id(text):
    tokens = text.split("_")
    prefix = "_".join(tokens[:-1])
    return prefix, int(tokens[-1])


def extract_unique_docs():
    unique_docs: Dict[str, str] = {}
    seen_docs: Set[str] = set()

    for n in range(1, 30):
        jsonl_path = get_rp_path("run6_voca_doc_map", f"{n}.jsonl")
        print(f"Processing {jsonl_path}")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                for line in f:
                    row = json.loads(line.strip())
                    doc_name = row['doc_name']
                    text = row['text']

                    # Only add if we haven't seen this doc_name before
                    if doc_name not in seen_docs:
                        unique_docs[doc_name] = text
                        seen_docs.add(doc_name)

    keys = list(unique_docs.keys())
    keys.sort(key=parse_data_id)

    # Save the unique document-text pairs
    output_path = get_rp_path("run6_unique_docs.jsonl")
    with open(output_path, 'w') as f_out:
        for doc_name in keys:
            text = unique_docs[doc_name]
            row = {"doc_name": doc_name, "text": text}
            f_out.write(json.dumps(row) + '\n')

    print(f"Found {len(unique_docs)} unique documents")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    extract_unique_docs()
