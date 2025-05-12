import json

from rule_gen.reddit.path_helper import get_j_res_save_path, get_split_subreddit_list


def swap_tuple_elements(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create a new list with swapped elements
    swapped_data = []
    for item in data:
        # Ensure we have a tuple with at least 4 elements
        if isinstance(item, list) and len(item) >= 4:
            # Create a new tuple with the 3rd and 4th elements swapped
            # Original: (str, int, list, str)
            # New: (str, int, str, list)
            if type(item[2]) == list and type(item[3]) == str:
                pass
            else:
                print(item)
                raise ValueError()
            swapped_item = (item[0], item[1], item[3], item[2])
            swapped_data.append(swapped_item)
        else:
            print(f"Warning: Skipping invalid item: {item}")

    # Write the swapped data to the output file
    with open(output_file, 'w') as f:
        # Convert tuples to lists for JSON serialization
        json_data = [list(item) for item in swapped_data]
        json.dump(json_data, f, indent=2)

    print(f"Processed {len(swapped_data)} items. Data saved to {output_file}")



def main():
    s9_run_name = "llama_s9nosb"
    print("load_run_name_fmt", s9_run_name)
    sb_list = get_split_subreddit_list("train")
    # sb_list = ["pics"]
    split = "val"
    for sb in sb_list:
        dataset = f"{sb}_2_{split}_100"
        in_path = get_j_res_save_path(s9_run_name, dataset)
        try:
            swap_tuple_elements(in_path, in_path)
        except ValueError:
            print(f"Warning: Skipping invalid item: {sb}")


if __name__ == "__main__":
    main()
