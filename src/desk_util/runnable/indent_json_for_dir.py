import os
import json
import fire


def indent_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                indented_json = json.dumps(data, indent=4, ensure_ascii=False)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(indented_json + '\n')
                print(f"Indented and updated: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")



if __name__ == "__main__":
    fire.Fire(indent_json_files)
