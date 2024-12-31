from desk_util.io_helper import read_csv_column, save_text_list_as_csv
from desk_util.path_helper import get_text_list_save_path


def ask_to_select_line(text):
    idx = 1
    line_d = {}
    for line in text.split("\n"):
        line = line.strip()
        if line:
            t = f"{idx}) {line}"
            line_d[idx] = line
            idx += 1
            print(t)
    sel = input("Select line: ")
    sel = int(sel)
    if sel == 0:
        out_line = ""
    else:
        out_line = line_d[sel]
    return out_line


def main():
    save_path = get_text_list_save_path("toxigen_head_100_para")
    text_list = read_csv_column(save_path, 0)
    text_list = text_list
    def do_iter():
        for text in text_list:
            out_line = ask_to_select_line(text)
            yield out_line

    save_path = get_text_list_save_path("toxigen_head_100_para_out")
    save_text_list_as_csv(do_iter(), save_path)


if __name__ == "__main__":
    main()
