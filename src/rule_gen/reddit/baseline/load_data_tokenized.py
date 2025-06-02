import tqdm
from krovetzstemmer import Stemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


# nltk.download('punkt', quiet=True)
def load_reddit_train_data(data_name, sb, role = "train") -> list[tuple[str, int]]:
    items = read_csv(get_reddit_train_data_path_ex(
        data_name, sb, role))

    output: list[tuple[str, int]] = []
    for text, label_s in items:
        output.append((text, int(label_s)))
    return output


def apply_tokenize(items: list[tuple[str, int]]) -> list[tuple[list[str], int]]:
    stemmer = Stemmer()
    output: list[tuple[list[str], int]] = []
    for text, label in tqdm(items):
        tokens = word_tokenize(text)
        tokens = [stemmer(t) for t in tokens]
        output.append((tokens, label))
    return output


def load_reddit_train_data2_tokenized(sb, role ="train") -> list[tuple[list[str], int]]:
    data_name = "train_data2"
    return load_reddit_train_data_tokenized(data_name, sb, role)


def load_reddit_train_data_tokenized(data_name, sb, role):
    data = load_reddit_train_data(data_name, sb, role)
    return apply_tokenize(data)