from rule_gen.reddit.dataset_build.wayback.do_download import get_reddit_archive_save_path
from rule_gen.reddit.path_helper import load_subreddit_list


def main():
    sb_names = load_subreddit_list()
    expected = " Dec 25, 2016 "
    for sb in sb_names:
        try:
            save_path = get_reddit_archive_save_path(sb)
            lines = open(save_path, "r").readlines()
            for l in lines:
                if "FILE ARCHIVED ON" in l:
                    if expected in l:
                        print(sb, "OK")
                    else:
                        print(sb, l)

        except FileNotFoundError as e:
            pass
        except Exception as e:
            print(e)



if __name__ == "__main__":
    main()
