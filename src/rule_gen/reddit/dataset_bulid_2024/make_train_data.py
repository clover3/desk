import random

from desk_util.io_helper import save_csv
from desk_util.path_helper import load_csv_dataset
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.path_helper import load_subreddit_list, get_reddit_train_data_path_ex


def convert_inf_data_to_train_format(dataset_name):
    payload = load_csv_dataset(dataset_name)
    labels = load_labels(dataset_name)
    labels_d = dict(labels)
    outputs = []
    for data_id, text in payload:
        e = text, labels_d[data_id]
        outputs.append(e)
    return outputs


def main():
    sb_list = load_subreddit_list()
    random.seed(42)
    success_list = []
    large_enough_list = []
    for sb in sb_list:
        try:
            dataset_name = f"{sb}_2024_2_train"
            data_name = "2024_2"
            save_path = get_reddit_train_data_path_ex(data_name, sb, "train")
            outputs = convert_inf_data_to_train_format(dataset_name)
            save_csv(outputs, save_path)
            if len(outputs) > 4000:
                large_enough_list.append(sb)
            print(sb, len(outputs))
            dataset_name = f"{sb}_2024_2_val"
            save_path = get_reddit_train_data_path_ex(data_name, sb, "val")
            outputs = convert_inf_data_to_train_format(dataset_name)
            save_csv(outputs, save_path)
            success_list.append(sb)
        except FileNotFoundError:
            pass

    print("success_list", success_list)
    print("large_enough_list", large_enough_list)


# success_list ['politics', 'AskReddit', 'science', 'worldnews', 'news', 'explainlikeimfive', 'relationships', 'TwoXChromosomes', 'gonewild', 'askscience', 'leagueoflegends', 'AskHistorians', 'Games', 'PoliticalDiscussion', 'personalfinance', 'aww', 'nosleep', 'CFB', 'pcmasterrace', 'pics', 'pokemongo', 'funny', 'GlobalOffensive', 'Futurology', 'MMA', 'europe', 'nfl', 'BlackPeopleTwitter', 'legaladvice', 'history', 'videos', 'AskWomen', 'sex', 'LateStageCapitalism', 'gaming', 'whatisthisthing', 'Showerthoughts', 'OutOfTheLoop', 'atheism', 'food', 'movies', 'india', 'books', 'depression', 'pokemon', 'nba', 'Christianity', 'anime', '2007scape', 'fantasyfootball', 'Overwatch', 'tifu', 'changemyview', 'space', 'conspiracy', 'canada', 'socialism', 'CanadaPolitics', 'nottheonion', 'gameofthrones', 'OldSchoolCool', 'AskTrumpSupporters', 'SuicideWatch', 'wow', 'LifeProTips', 'SubredditDrama', 'technology', 'TheSilphRoad', 'me_irl', 'IAmA', 'DestinyTheGame', 'television', 'PurplePillDebate', 'asoiaf', 'NeutralPolitics']
# large_enough_list ['AskReddit', 'worldnews', 'news', 'gonewild', 'pics', 'legaladvice', 'Christianity', 'conspiracy', 'SuicideWatch']
if __name__ == '__main__':
    main()
