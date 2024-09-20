import os
import json
from collections import Counter

from toxicity.cpath import output_root_path

def get_key_of_max_value(dictionary):
    if not dictionary:
        return None
    return max(dictionary, key=dictionary.get)


def merge_json_files(directory):
    merged_dict = {}

    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Read and parse each JSON file
            with open(file_path, 'r') as file:
                try:
                    file_data = json.load(file)

                    # Merge the file data into the merged_dict
                    merged_dict.update(file_data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    return merged_dict


def main():
    # Example usage
    directory_path = os.path.join(output_root_path, "grad_mag")
    result = merge_json_files(directory_path)

    mlp_only = {k: v for k, v in result.items() if "mlp" in k}
    down_proj_only = {k: v for k, v in result.items() if "down_proj" in k}
    todo = [
        ("result", result),
        ("mlp_only", mlp_only),
        ("down_proj_only", down_proj_only)
    ]
    for name, cands in todo:
        print(name)
        cands = Counter(cands)
        for k, v in cands.most_common(10):
            print("{0}\t{1:.3f}".format(k, v))
        print()


if __name__ == "__main__":
    main()