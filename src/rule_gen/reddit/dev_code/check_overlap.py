from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path

"""
No Spamming.
No Trolling.
No Racism/Anti-Semitism.
No Releasing Personal Information or Doxxing.
No Vote Manipulation, Brigading, or Asking for Votes.
No Dissenters or SJWs.
No Posts Related to Being Banned from Other Subreddits.
No Posts About Subreddit Suggestions or Concerns (Use Modmail Instead).
No Posts About Trump Assassination Threats (Send screenshots and an Archive.is link to the FBI).
AfterBerners (Former BernieBots) MUST c .
"""

def main():
    subreddit = "The_Donald"
    p = get_reddit_train_data_path(subreddit, "train")
    items = read_csv(p)
    pos_set = {t for t, l in items if l == "1"}
    neg_set = {t for t, l in items if l == "0"}

    intersect = pos_set.intersection(neg_set)
    print(len(intersect))

    for t in intersect:
        print(t)

    return NotImplemented


if __name__ == "__main__":
    main()