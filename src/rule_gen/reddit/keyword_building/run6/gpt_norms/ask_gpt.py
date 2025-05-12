import tqdm
import json

from desk_util.io_helper import read_jsonl
from rule_gen.reddit.keyword_building.run6.corpus_based_analysis.unique_comment import parse_data_id
from rule_gen.reddit.path_helper import get_rp_path


def get_gpt4o_request_fn(model_name):
    from desk_util.open_ai import OpenAIChatClient
    client = OpenAIChatClient(model_name)
    inst_fmt = "If the following comment is posted in {} subreddit and moderated (deleted), what would be the reason?"
    def predict(data_id, text):
        sb, _ = parse_data_id(data_id)
        instruction = inst_fmt.format(sb)
        prompt = instruction + "<text> {} </text>".format(text)
        return client.request(prompt)
    return predict


def main():
    model_name = "gpt-4o"
    output_path = get_rp_path("run6_unique_docs.jsonl")
    data = read_jsonl(output_path)
    request_fn = get_gpt4o_request_fn(model_name)
    output_path = get_rp_path(f"run6_unique_docs_why_{model_name}.jsonl")
    with open(output_path, 'w') as f_out:
        for e in tqdm.tqdm(data):
            ret_text = request_fn(e["doc_name"], e["text"])
            row = {"doc_name": e["doc_name"], "text": ret_text}
            f_out.write(json.dumps(row) + '\n')
            f_out.flush()


if __name__ == "__main__":
    main()