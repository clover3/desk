from desk_util.io_helper import save_text_list_as_csv, read_csv_column
from desk_util.path_helper import get_text_list_save_path


def main():
    def enum_paras():
        for i in range(10):
            save_path = get_text_list_save_path(f"toxigen_train_para_fold_{i}")
            yield from read_csv_column(save_path, 0)

    text_itr = enum_paras()
    output = parse_para(text_itr)
    save_path = get_text_list_save_path(f"toxigen_train_para_all_fold_selected")
    save_text_list_as_csv(output, save_path)


def parse_para(text_itr):
    output = []
    for idx, item in enumerate(text_itr):
        print("idx=", idx)
        lines = item.split("\n")
        lines = [l.strip() for l in lines if l.strip()]
        selected_line = None
        for line_idx, line in enumerate(lines):
            line_head_keys = ["Paraphrased text:",
                              "Paraphrased sentence:",
                              "rewritten sentence:",
                              ]
            for key in line_head_keys:
                if line.lower().startswith(key.lower()):
                    if len(line) > len(key) + 10:
                        selected_line = line[len(key):]
                    else:
                        if line_idx + 1 < len(lines):
                            selected_line = lines[line_idx + 1]
                if line.lower().endswith(key.lower()):
                    if line_idx + 1 < len(lines):
                        selected_line = lines[line_idx + 1]

        if selected_line is None:
            line_keys = ["I've selected",
                         "I'll select",
                         "I'll choose",
                         "I've chosen",
                         "Let's replace",
                         "Let's select",
                         "I selected ",
                         "Here's a paraphrased",
                         "Here is a paraphrased"
                         ]

            for line_idx, line in enumerate(lines):
                for key in line_keys:
                    if key in line:
                        if line_idx + 1 < len(lines):
                            selected_line = lines[line_idx + 1]
                            break
        warn_keys = ["select", "replace", "paraphrase"]
        show_parsing = False
        if selected_line is None:
            show_parsing = True
        else:
            for key in warn_keys:
                if key in selected_line:
                    print("Warning", key)
                    show_parsing = True

        if show_parsing == True:
            for line_idx, line in enumerate(lines):
                print(f"{line_idx}: {line}")
            print("Selected line:", selected_line)
            print("=====")

        output.append(selected_line.strip().strip("\""))
    return output


if __name__ == "__main__":
    main()
