from collections import Counter
import math
from toxicity.io_helper import read_csv_column, save_text_list_as_csv
from toxicity.llama_helper.lf_client import transform_text_by_llm
from toxicity.misc_lib import get_second
from toxicity.para_gen.line_selector import ask_to_select_line
from toxicity.path_helper import get_toxigen_failure_save_path, get_text_list_save_path
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')


from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import read_csv_column


def bag_of_words_similarity(sentence1, sentence2):
    # Tokenize the sentences into words
    words1 = word_tokenize(sentence1.lower())
    words2 = word_tokenize(sentence2.lower())

    # Create bag of words representations
    bag1 = Counter(words1)
    bag2 = Counter(words2)

    # Get the union of all words
    all_words = set(bag1.keys()) | set(bag2.keys())

    # Calculate dot product
    dot_product = sum(bag1.get(word, 0) * bag2.get(word, 0) for word in all_words)

    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(count **2 for count in bag1.values()))
    magnitude2 = math.sqrt(sum(count **2 for count in bag2.values()))

    # Compute cosine similarity
    if magnitude1 * magnitude2 == 0:
        return 0  # Handle division by zero
    else:
        return dot_product / (magnitude1 * magnitude2)


def main():
    save_path = get_text_list_save_path("toxigen_head_100_para")
    para_list = read_csv_column(save_path, 0)
    n_item = 100
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    test_dataset = list(test_dataset)[:n_item]
    text_list = [e["text"] for e in test_dataset]

    outputs = []
    for orig, response in zip(text_list, para_list):
        lines = response.split("\n")
        lines = [t.strip() for t in lines if t.strip()]
        lines = [l for l in lines if orig not in l]
        sims = [bag_of_words_similarity(orig, l) for l in lines]

        idx_sims = [(i, s) for i, s in enumerate(sims)]
        idx_sims.sort(key=get_second, reverse=True)

        if len([s for s in sims if s > 0.3]) >= 1:
            para_idx = idx_sims[0][0]
        else:
            para_idx = None
            print(lines)
            print(sims)

        if para_idx is not None:
            para = lines[para_idx]
        else:
            para = "FAIL"

        outputs.append(para)

    save_path = get_text_list_save_path("toxigen_head_100_para_out")
    save_text_list_as_csv(outputs, save_path)



if __name__ == "__main__":
    main()
